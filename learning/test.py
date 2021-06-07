import os
import argparse
import pickle5 as pickle
import numpy as np
import shelve
import yaml
import ray
from ray.rllib.evaluation.worker_set import WorkerSet

import misc
from policies import PolicyManager
from envs import wrappers
from trainers import get_trainer_class
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Roll out a reinforcement learning agent given a checkpoint.')
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Checkpoint from which to roll out.')
    parser.add_argument( # TODO: this should be inside config but it's not stored
        '--run',
        type=str,
        default='PPO',
        help='The algorithm or model to train')
    parser.add_argument(
        '--eval-config',
        type=str,
        default=None,
        help='Config for evaluation (Overwrite train config).')
    parser.add_argument(
        '--save-rollout',
        default=False,
        action='store_true',
        help='Whether to save rollout.')
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Path to saving rollout. Default to checkpoint directory.')
    parser.add_argument(
        '--save-dir-suffix',
        type=str,
        default=None,
        help='Suffix to result directory.')
    parser.add_argument(
        '--monitor',
        default=False,
        action='store_true',
        help='Use gym Monitor-like wrapper to record video.')
    parser.add_argument(
        '--episodes',
        default=1,
        type=int,
        help='Number of complete episodes to roll out.')
    parser.add_argument(
        '--save-all',
        default=False,
        action='store_true',
        help='Save all data generated by the step() method.')
    parser.add_argument(
        '--num-workers',
        default=0,
        type=int,
        help='Number of workers.')
    parser.add_argument(
        '--num-gpus',
        default=1,
        type=int,
        help='Number of GPUs.')
    parser.add_argument(
        '--temp-dir',
        default='~/tmp',
        type=str,
        help='Directory for temporary files generated by ray.')
    parser.add_argument(
        '--local-mode',
        action='store_true',
        help='Whether to run ray with `local_mode=True`. ')
    parser.add_argument(
        '--task-mode',
        default=None,
        type=str,
        help='VISTA task mode (for obstacle avoidance and takeover).')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic action.')
    parser.add_argument(
        '--use-shelve',
        default=False,
        action="store_true",
        help='Save rollouts into a python shelf file (will save each episode '
        'as it is generated). An output filename must be set using --out.')
    parser.add_argument(
        '--track-progress',
        default=False,
        action='store_true',
        help='Write progress to a temporary file (updated '
        'after each episode). An output filename must be set using --out; '
        'the progress file will live in the same folder.')

    args = parser.parse_args()

    return args


def main():
    # Load config
    args = parse_args()
    if os.path.isdir(args.checkpoint):
        args.checkpoint = misc.get_latest_checkpoint(args.checkpoint)
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, 'params.pkl')
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, '../params.pkl')
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)

    # Overwrite some config with arguments
    config['num_workers'] = args.num_workers
    config['num_gpus'] = args.num_gpus
    if args.task_mode is not None:
        config['env_config']['task_mode'] = args.task_mode

    if args.eval_config:
        eval_config = misc.load_yaml(args.eval_config)
        config = misc.update_dict(config, eval_config)
    else:
        eval_config = config

    # Register custom model
    misc.register_custom_env(config['env'])
    misc.register_custom_model(config['model'])

    # Start ray
    args.temp_dir = os.path.abspath(os.path.expanduser(args.temp_dir))
    ray.init(
        local_mode=args.local_mode,
        _temp_dir=args.temp_dir,
        include_dashboard=False)

    # Get agent
    cls = get_trainer_class(args.run)
    agent = cls(env=config['env'], config=config)
    if args.checkpoint:
        agent.restore(args.checkpoint)

    # Evaluation
    if args.save_rollout: 
        if not args.out:
            save_dir_name = 'results'
            if args.save_dir_suffix:
                save_dir_name = save_dir_name + '_{}'.format(args.save_dir_suffix)
            save_dir = os.path.join(os.path.expanduser(config_dir), save_dir_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            args.out = os.path.join(save_dir, 'rollout.pkl')

        # save evaluation config
        config_path = os.path.splitext(args.out)[0] + '_eval_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(eval_config, f)
    with RolloutSaver(
            args.out,
            args.use_shelve,
            write_update_file=args.track_progress,
            target_steps=None,
            target_episodes=args.episodes,
            save_all=args.save_all) as saver:
        test(args, agent, args.episodes, saver)

    # End ray
    ray.shutdown()


def test(args, agent, num_episodes, saver):
    assert hasattr(agent, 'workers') and isinstance(agent.workers, WorkerSet)

    env = agent.workers.local_worker().env
    if args.monitor:
        if saver._outfile:
            save_dir = os.path.dirname(saver._outfile)
        else:
            save_dir = os.path.dirname(args.checkpoint)
        video_dir = os.path.join(save_dir, 'monitor')
        env = wrappers.MultiAgentMonitor(env, video_dir, video_callable=lambda x: True, force=True)
    policy_mapping_fn = agent.config["multiagent"]["policy_mapping_fn"]
    policy_map = agent.workers.local_worker().policy_map

    for ep in range(num_episodes):
        if args.save_rollout:
            saver.begin_rollout()
        obs = env.reset()
        done = False
        episode_reward = dict()
        for agent_id in obs.keys():
            episode_reward[agent_id] = 0.
        state = {p: m.get_initial_state() for p, m in policy_map.items()}
        has_state = {p: len(s) > 0 for p, s in state.items()}
        while not done:
            act = dict()
            for agent_id, a_obs in obs.items():
                policy_id = policy_mapping_fn(agent_id)
                if has_state[policy_id]:
                    a_state = state[policy_id]
                    a_act, a_state, _ = agent.compute_action(a_obs, a_state, policy_id=policy_id, explore=args.deterministic)
                    state[agent_id] = a_state
                else:
                    a_act = agent.compute_action(a_obs, policy_id=policy_id, explore=args.deterministic)
                act[agent_id] = a_act
            next_obs, rew, done, info = env.step(act)
        
            # save data
            if args.save_rollout:
                saver.append_step(obs, act, next_obs, rew, done, info)

            for agent_id, a_rew in rew.items():
                episode_reward[agent_id] += a_rew

            done = np.any(list(done.values()))
            obs = next_obs
        if args.save_rollout:
            saver.end_rollout()
        print("Episode #{}: reward: {}".format(ep, episode_reward))


class RolloutSaver:
    """Utility class for storing rollouts.
    Currently supports two behaviours: the original, which
    simply dumps everything to a pickle file once complete,
    and a mode which stores each rollout as an entry in a Python
    shelf db file. The latter mode is more robust to memory problems
    or crashes part-way through the rollout generation. Each rollout
    is stored with a key based on the episode number (0-indexed),
    and the number of episodes is stored with the key "num_episodes",
    so to load the shelf file, use something like:
    with shelve.open('rollouts.pkl') as rollouts:
       for episode_index in range(rollouts["num_episodes"]):
          rollout = rollouts[str(episode_index)]
    If outfile is None, this class does nothing.
    """

    def __init__(self,
                 outfile=None,
                 use_shelve=False,
                 write_update_file=False,
                 target_steps=None,
                 target_episodes=None,
                 save_all=False):
        self._outfile = outfile
        self._update_file = None
        self._use_shelve = use_shelve
        self._write_update_file = write_update_file
        self._shelf = None
        self._num_episodes = 0
        self._rollouts = []
        self._current_rollout = []
        self._total_steps = 0
        self._target_episodes = target_episodes
        self._target_steps = target_steps
        self._save_all = save_all

    def _get_tmp_progress_filename(self):
        fname = '__progress_{}.txt'.format(os.path.basename(self._outfile).replace('.','_'))
        return os.path.join(os.path.dirname(self._outfile), fname)

    @property
    def outfile(self):
        return self._outfile

    def __enter__(self):
        if self._outfile:
            if self._use_shelve:
                # Open a shelf file to store each rollout as they come in
                self._shelf = shelve.open(self._outfile)
            else:
                # Original behaviour - keep all rollouts in memory and save
                # them all at the end.
                # But check we can actually write to the outfile before going
                # through the effort of generating the rollouts:
                try:
                    with open(self._outfile, "wb") as _:
                        pass
                except IOError as x:
                    print("Can not open {} for writing - cancelling rollouts.".
                          format(self._outfile))
                    raise x
            if self._write_update_file:
                # Open a file to track rollout progress:
                self._update_file = open(self._get_tmp_progress_filename(), 'w')
        return self

    def __exit__(self, type, value, traceback):
        if self._shelf:
            # Close the shelf file, and store the number of episodes for ease
            self._shelf["num_episodes"] = self._num_episodes
            self._shelf.close()
        elif self._outfile and not self._use_shelve:
            # Dump everything as one big pickle:
            pickle.dump(self._rollouts, open(self._outfile, "wb"))
        if self._update_file:
            # Remove the temp progress file:
            self._update_file.close()
            self._update_file = None

    def _get_progress(self):
        if self._target_episodes:
            return "{} / {} episodes completed".format(self._num_episodes,
                                                       self._target_episodes)
        elif self._target_steps:
            return "{} / {} steps completed".format(self._total_steps,
                                                    self._target_steps)
        else:
            return "{} episodes completed".format(self._num_episodes)

    def begin_rollout(self):
        self._current_rollout = []

    def end_rollout(self):
        if self._outfile:
            if self._use_shelve:
                # Save this episode as a new entry in the shelf database,
                # using the episode number as the key.
                self._shelf[str(self._num_episodes)] = self._current_rollout
            else:
                # Append this rollout to our list, to save laer.
                self._rollouts.append(self._current_rollout)
        self._num_episodes += 1
        if self._update_file:
            self._update_file.seek(0)
            self._update_file.write(self._get_progress() + "\n")
            self._update_file.flush()

    def append_step(self, obs, action, next_obs, reward, done, info):
        """Add a step to the current rollout, if we are saving them"""
        if self._outfile:
            if self._save_all:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done, info])
            else:
                self._current_rollout.append(
                    [reward, done, info])
        self._total_steps += 1


if __name__ == '__main__':
    main()