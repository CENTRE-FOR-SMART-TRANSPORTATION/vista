from pathlib import Path
import os
import numpy as np
import pandas as pd
import argparse
import tkinter as tk
import tkinter.filedialog
import open3d as o3d
import laspy
import sys
import os
from tkinter import Tk
from pathlib import Path
from tqdm import tqdm

DESKTOP = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') 

# Obtain the path to our scenes
def obtain_scene_path() -> str:
    """Obtains the path to the folder containing all of the outputs
    to the Vista simulator.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        scenes_folderpath (str): Path to the folder containing the Vista outputs.
    """

    # Manually open trajectory folder
    Tk().withdraw()
    scenes_folderpath = tk.filedialog.askdirectory(
        initialdir=DESKTOP, title="Please select the Vista output folder"
    )
    print(
        f"\nYou have chosen to open the folder to the scenes:\n{scenes_folderpath}"
    )

    num_scenes = len(
        [
            name
            for name in os.listdir(scenes_folderpath)
            if os.path.isfile(os.path.join(scenes_folderpath, name))
        ]
    )
    print(f"{num_scenes} scenes were found for the corresponding road section folder.")

    return os.path.abspath(scenes_folderpath)

# Obtain the path to the sensor config
def obtain_sensor_path() -> str:
    """Opens the sensor configuration file from command-line argument or
    through UI.

    Args:
        args (argparse.Namespace): Contains the command-line arguments.

    Returns:
        sensorcon_filepath (str): Path to the sensor configuration
    """

    # Manually get sensor configuration file
    Tk().withdraw()
    sensorcon_filepath = tk.filedialog.askopenfilename(
        filetypes=[(".json files", "*.json"), ("All files", "*")],
        initialdir=os.path.join(DESKTOP),
        title="Please select the sensor configuration file",
    )
    print(f"\nYou have chosen to open the sensor file:\n{sensorcon_filepath}")

    return os.path.abspath(sensorcon_filepath)


# Obtain the trajectory
def obtain_trajectory_path() -> str:
    """Obtains a pregenerated trajectory and reads each of them into
    a container class.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Trajectory: Container class for our imported trajectory data.
    """

    # Manually open trajectory folder
    Tk().withdraw()
    trajectory_folderpath = tk.filedialog.askdirectory(
        initialdir=DESKTOP, title="Please select the trajectory folder"
    )
    print(
        f"You have chosen to open the trajectory folder:\n{trajectory_folderpath}"
    )


    # Read the filenames of the trajectories into a list
    trajectory_files = [
        path
        for path in os.listdir(trajectory_folderpath)
        if os.path.isfile(os.path.join(trajectory_folderpath, path))
    ]

    # Sanity check
    # if len(trajectory_files) != 5:
    #  raise(RuntimeError(f"Trajectory folder is missing files!\nExpected count: 5 (got {len(trajectory_files)})!"))
    assert (
        len(trajectory_files) == 5
    ), f"Trajectory folder is missing files!\nExpected count: 5 (got {len(trajectory_files)})!"


    return os.path.abspath(trajectory_folderpath)

# Open our .las point cloud into memory
def obtain_las_path() -> str:
    """
    Opens a .las file when prompted to do so. Can force a predetermined filename
    (default called as None for manual input)

    Arguments:
    verbose (bool): Setting to print extra information to the command line.

    predetermined_filename (string): The predetermined file name of the point cloud.
    User can be manually prompted to enter the point cloud, or it can be set to some
    point cloud via command line for automation. See main() for command line syntax.
    """
    # Manually obtain file via UI
    Tk().withdraw()
    las_filename = tk.filedialog.askopenfilename(
        filetypes=[(".las files", "*.las"), ("All files", "*")],
        initialdir=DESKTOP,
        title="Please select the main point cloud",
    )

    print(f"You have chosen to open the point cloud:\n{las_filename}")

    return os.path.abspath(las_filename)

scenes = obtain_scene_path()
config = obtain_sensor_path()
trajectory = obtain_trajectory_path()
las_file = obtain_las_path()

env_str = f'''export SCENES={scenes}
export TRAJECTORY={trajectory}
export CONFIG={config}
export INPUT={las_file}
'''

with open(".env", "w") as f:
    f.write(env_str)