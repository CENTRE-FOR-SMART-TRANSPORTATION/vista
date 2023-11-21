import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import tkinter as tk
import sys
import os
import glob
import time
import pickle

from tkinter import Tk
from pathlib import Path
from tqdm import tqdm
import utils

import matplotlib
from classes import SensorConfig, Trajectory

import file_tools
import sensorpoints
import argparse
from dotenv import load_dotenv
import shutil
import math

ZOOM = 0.07
VIEW = "isometric-constant"
FRONT_X = -1
FRONT_Y = 0
FRONT_Z = 1

def render_sensor_fov(
    cfg: SensorConfig,
    traj: Trajectory,
    road: o3d.geometry.PointCloud,
    src_name: str,
    sensor_images_path: str,
    screen_width: int,
    screen_height: int,
    offset: int
) -> None:
    global VIEW, ZOOM

    # Helper function to set the visualizer POV
    def set_visualizer_pov(mode: str) -> None:
        if mode == "front":
            x1, y1, z1 = traj.getForwards()[frame, :]
            x2, y2, z2 = traj.getUpwards()[frame, :]
            x3, y3, z3 = traj.getObserverPoints()[frame, :]
            z1 = 0

            ctr.set_front([-1*x1, -1*y1, z1])
            ctr.set_up([x2, y2, z2])
            ctr.set_lookat([x3, y3, z3+1.8])
            ctr.set_zoom(ZOOM)
        elif mode == "isometric":
            x1, y1, z1 = traj.getForwards()[frame, :]
            x2, y2, z2 = traj.getUpwards()[frame, :]
            x3, y3, z3 = traj.getRoadPoints()[frame, :]
            z1 = 0
            x1, y1 = -x1, -y1

            rotation_z = math.radians(45)  # degrees left
            rotation_x = math.radians(30)  # degrees downwards

            # Rotation matrices
            rotation_z_matrix = np.array([
                [np.cos(rotation_z), -np.sin(rotation_z), 0],
                [np.sin(rotation_z), np.cos(rotation_z), 0],
                [0, 0, 1]
            ])

            rotation_x_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(rotation_x), -np.sin(rotation_x)],
                [0, np.sin(rotation_x), np.cos(rotation_x)]
            ])

            # Apply rotations
            forwards = np.dot(rotation_x_matrix, np.dot(
                rotation_z_matrix, np.array([x1, y1, z1])))
            upwards = np.dot(rotation_x_matrix, np.dot(
                rotation_z_matrix, np.array([x2, y2, z2])))
                
            # Center the view around the sensor FOV
            ctr.set_lookat(traj.getRoadPoints()[frame, :])
            ctr.set_zoom(ZOOM)
        elif mode == "isometric-constant":
            ctr.set_front([FRONT_X, FRONT_Y, FRONT_Z])
            ctr.set_up([0, 0, 1])

            # Center the view around the sensor FOV
            ctr.set_lookat(traj.getRoadPoints()[frame, :])

            ctr.set_zoom(ZOOM)

    # Setup our visualizer
    vis = o3d.visualization.Visualizer()
    # NOTE open3d.visualization.rendering.OffscreenRenderer can probably be used here
    # instead of calling a GUI visualizer
    vis.create_window(window_name=f"Replay of sensor FOVs on {src_name}",
                      width=screen_width,
                      height=screen_height
                      )

    vis.set_full_screen(True)  # Full screen to capture full view

    # Obtain view control of the visualizer to change POV on setup
    # NOTE Currently, as of 5/19/2023, the get_view_control() method for the open3d.Visualizer class
    # only returns a copy of the view control as opposed to a reference.
    if (o3d.__version__ == "0.17.0"):
        pass
    else:
        ctr = vis.get_view_control()

    # Configure our render option
    render_opt = vis.get_render_option()
    render_opt.point_size = 1.0
    render_opt.show_coordinate_frame = True  # Does this even work
    render_opt.background_color = np.array(
        [16/255, 16/255, 16/255])  # 8-bit RGB, (16, 16, 16)

    # Initalize geometries
    vis.add_geometry(road)

    vis.poll_events()
    vis.update_renderer()

    # Begin our replay of the sensor FOV
    num_points = traj.getNumPoints()
    fov_points = utils.generate_sensor_points(cfg)
    if not os.path.exists(sensor_images_path):
        os.makedirs(sensor_images_path)

    for frame in range(0+offset, num_points-offset):
        # Set the view to look at the next road point
        set_visualizer_pov(VIEW)
        vis.update_geometry(road)
        vis.poll_events()
        vis.update_renderer()

        # Save rendered scene to an image so that we can write it to a video
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        img = (img[:, :]*255).astype(np.uint8)  # Normalize RGB to 8-bit

        cv2.imwrite(filename=os.path.join(
            sensor_images_path, f"{frame-offset}.png"), img=img)

    print(f"\nFOV rendering on road complete.")
    # vis.clear_geometries()
    vis.destroy_window()

    return

# Obtains screen size (width, height) in pixels


def obtain_screen_size() -> tuple:
    # Obtain screen parameters for our video
    from tkinter import Tk
    root = Tk()
    root.withdraw()

    SCREEN_WIDTH, SCREEN_HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()

    return (SCREEN_WIDTH, SCREEN_HEIGHT)


def check_for_padded(path_to_scenes: str) -> int:
    path_to_scenes_ext = os.path.join(path_to_scenes, '*.txt')
    filenames = [os.path.basename(abs_path)
                 for abs_path in glob.glob(path_to_scenes_ext)]
    offset = int(min(filenames, key=lambda x: int(
        (x.split('_'))[1])).split('_')[1])

    return offset


def main():
    # Parse our command line arguments
    def parse_cmdline_args() -> argparse.Namespace:
        # use argparse to parse arguments from the command line
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--config", type=str, default=None, help="Path to sensor config file"
        )
        parser.add_argument(
            "--trajectory", type=str, default=None, help="Path to trajectory folder"
        )
        parser.add_argument(
            "--observer_height", type=float, default=1.8, help="Height of the observer in m"
        )
        parser.add_argument(
            "--scenes", type=str, default=None, help="Path to the Vista output folder"
        )
        parser.add_argument(
            "--numScenes", type=int, default=1, help="Number of Vista output folders"
        )

        parser.add_argument("--x", type=float, default=-1, help="x coord of front vector")

        parser.add_argument("--y", type=float, default=0, help="y coord of front vector")

        parser.add_argument("--z", type=float, default=1, help="z coord of front vector")

        parser.add_argument("--input", type=str, default=None,
                            help="Path to the .las file")

        parser.add_argument("--zoom", type=float, default=0.3,
                            help="Zoom level of sensor fov")

        parser.add_argument("--view", type=str, default="isometric",
                            help="Option for the view of the points", choices=["front", "isometric", "isometric-constant"])

        return parser.parse_args()

    args = parse_cmdline_args()
    load_dotenv()

    args.scenes = os.environ["SCENES"] if args.scenes is None else args.scenes
    args.trajectory = os.environ["TRAJECTORY"] if args.trajectory is None else args.trajectory
    args.config = os.environ["CONFIG"] if args.config is None else args.config
    args.input = os.environ["INPUT"] if args.input is None else args.input

    path_to_scenes = file_tools.obtain_scene_path(args)
    traj = file_tools.obtain_trajectory_details(args)
    cfg = sensorpoints.open_sensor_config_file(args)
    road = file_tools.open_las(args)

    sensor_images_path = os.path.join(os.getcwd(), "las_pov/")
    if os.path.exists(sensor_images_path):
        shutil.rmtree(sensor_images_path)
        os.makedirs(sensor_images_path)

    # getting the required objects for crating the video
    screen_wh = obtain_screen_size()
    frame_offset = check_for_padded(path_to_scenes)

    road_o3d, src_name = utils.las2o3d_pcd(road)

    global ZOOM, VIEW, FRONT_X, FRONT_Y, FRONT_Z
    ZOOM = args.zoom if args.zoom is not None else ZOOM
    VIEW = args.view if args.view is not None else VIEW
    FRONT_X = args.x if args.x is not None else FRONT_X
    FRONT_Y = args.y if args.y is not None else FRONT_Y
    FRONT_Z = args.z if args.z is not None else FRONT_Z

    render_sensor_fov(cfg=cfg,
                      traj=traj,
                      road=road_o3d,
                      src_name=src_name,
                      sensor_images_path=sensor_images_path,
                      screen_width=screen_wh[0],
                      screen_height=screen_wh[1],
                      offset=frame_offset
                      )


if __name__ == "__main__":
    main()
