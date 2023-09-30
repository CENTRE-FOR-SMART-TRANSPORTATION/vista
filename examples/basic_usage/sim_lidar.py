import argparse
import numpy as np
import os
import cv2

import vista


def main(args):
    world = vista.World(
        args.trace_path, trace_config={"road_width": 4, "master_sensor": "lidar_3d"}
    )
    car = world.spawn_agent(
        config={
            "length": 5.0,
            "width": 2.0,
            "wheel_base": 2.78,
            "steering_ratio": 14.7,
            "lookahead_road": True,
        }
    )
    lidar_config = {
        "yaw_fov": (args.yaw_min, args.yaw_max),
        "pitch_fov": (args.pitch_min, args.pitch_max),
        "frame": args.frame,
        "yaw_res": args.resolution_hz,
        "pitch_res": args.resolution_vl,
        "downsample": args.downsample,
        "culling_r": args.culling_r,
        "roadsection_filename": args.filename
    }
    lidar = car.spawn_lidar(lidar_config)
    display = vista.Display(world)

    world.reset()
    display.reset()

    # while not car.done:
    action = follow_human_trajectory(car)
    # car.step_dynamics(action)
    # car.step_sensors()

    # vis_img = display.render()
    # cv2.imshow("Visualize LiDAR", vis_img[:, :, ::-1])
    # cv2.waitKey(10000)


def follow_human_trajectory(agent):
    action = np.array(
        [agent.trace.f_curvature(agent.timestamp), agent.trace.f_speed(agent.timestamp)]
    )
    return action


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description="Run the simulator with random actions"
    )
    parser.add_argument(
        "--trace-path",
        type=str,
        nargs="+",
        help="Path to the traces to use for simulation",
    )
    parser.add_argument(
        "--resolution_hz",
        type=float,
        default=0.11,
        help="Output resolution_hz",
    )
    parser.add_argument(
        "--resolution_vl",
        type=float,
        default=0.11,
        help="Output resolution_vl",
    )
    parser.add_argument(
        "--yaw-min",
        type=float,
        default=-180,
        help="Minimum yaw angle",
    )
    parser.add_argument(
        "--yaw-max",
        type=float,
        default=180,
        help="Maximum yaw angle",
    )
    parser.add_argument(
        "--pitch-min",
        type=float,
        default=-21,
        help="Minimum pitch angle",
    )
    parser.add_argument(
        "--pitch-max",
        type=float,
        default=19,
        help="Maximum pitch angle",
    )
    parser.add_argument("--culling-r", type=float, default=1, help="Culling Rate")
    parser.add_argument("--frame", type=int, help="Frame number")
    parser.add_argument("--downsample", action="store_true")
    parser.add_argument("--filename", type=str, help="Filename of the .las file")

    args = parser.parse_args()

    main(args)
