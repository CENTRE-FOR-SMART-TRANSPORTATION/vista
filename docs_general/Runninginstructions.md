# Running instructions for this fork of the Vista simulator

[Back to README.md](README.md#process-for-generating-vista-outputs)

## Dependencies

- A Linux environment
- ``nvidia-cuda-toolkit`` (I need to verify this though)
- The other Python requirements should be in the ``vista_venv`` folder within this repository
  - _Type ``source vista_venv/bin/activate`` to activate the virtual environment._

## Overview

In order to generate the scenes, we use processes (i.e. shell scripts) controlled by a process controller (the main shell script itself) for speed.

There are two implementations for the process controller, (``process_controller.sh``/``process_controller_python.sh`` and ``process_controller_v2.sh``, all of which are located at ``vista/examples/processes/single/``), where we will go into detail with them later on.

Currently, the max is around 6 processes (haven't tested more than 6) on an Alienware Aurora R13 with an i9-12900KF and an RTX 3080.

- ``process_controller.sh`` & ``process_controller_python.sh`` **(legacy)**: Uses ``xterm`` to host each process, and is a bit cleaner and easier to use. Some processes may crash at times, which is why I decided to make a new implementation of it.
  - The difference between the two is that graphs will be generated using MATLAB, and Python, respectively. Both function the same otherwise.

- ``process_controller_v2.sh``: Uses subshells to host each process, and is expandable to support some number of processes depending on hardware. Also has some more features, such as identifying any missing Vista outputs upon completion, and skipping trajectory generation if a trajectory already exists to save time.
  - Note: Unlike the other process controllers, you can't simply kill all of the processes by pressing CTRL+C with version 2 since the processes are in the background. In order to kill the processes, we input the following in another terminal instance:

  ```bash
  # Kills the processes as well as the process controller
  $ bash kill_processes.sh
  ```

## Usage

As both versions of the process controllers are shell scripts, you run them by typing one of the following, from the root directory of the repository:

```bash
# Uses MATLAB for our graphical outputs
$ bash examples/processes/single/process_controller.sh

# Uses Python for our graphical outputs
$ bash examples/processes/single/process_controller_python.sh

# Uses MATLAB/Python for our graphical outputs (specified within the shell script)
$ bash examples/processes/single/process_controller_v2.sh
```

But before you go off and run these shell scripts, first of all, you first have to configure the parameters needed, such as:

- The number of processes that we will use to generate the outputs in parallel.
- The road section that you want to perform a Vista simulation on.
- The .json file detailing the sensor configuration that we will use.
  - *You would also have to manually input the sensor's parameters for now, I will find a way to parse the .json file automatically into the shell script.*
- The observer height, or height that we would want to place our sensor at.
- The option to pad outputs (we should always pad our generation to save some time, more on this later)
- Whether or not to generate our outputs using MATLAB or Python.
- The start frame and end frame defining the range of the road that we will perform our simulation on.
  - *You can manually override the default range if you wish to generate outputs for a specific range.*
