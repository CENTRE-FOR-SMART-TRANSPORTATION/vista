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
import argparse
import shutil

ZOOM = 0.03
VIEW = "isometric"

class PointCloudOpener:
    # Opens one specified point cloud as a Open3D tensor point cloud for parallelism
    def open_point_cloud(
        self, path_to_scenes: str, frame: int, res: np.float32, mode: str
    ) -> o3d.t.geometry.PointCloud:
        """Reads a specified point cloud from a path into memory.
        This is called in the parallelized loop in obtain_scenes().

        Args:
            path2scenes (str): The path to the folder containing the scenes.
            frame (int): The frame of the particular scene.
            res (np.float32): The resolution of the sensor at which the scene was recorded.
            This should be given in the filename, where scene names are guaranteed to be
            "output_<FRAME>_<RES>.txt".

        Returns:
            pcd (o3d.t.geometry.PointCloud): Our point cloud, in tensor format.
        """

        scene_name = f"output_{frame}_{res:.2f}.txt"
        path_to_scene = os.path.join(path_to_scenes, scene_name)
        # print(scene_name)
        
        # Skip our header, and read only XYZ coordinates
        df = pd.read_csv(path_to_scene, skiprows=0, usecols=[0, 1, 2])
        xyz = df.to_numpy() / 1000
        
        # Create Open3D point cloud object with tensor values.
        # For parallelization, outputs must be able to be serialized 
        pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
        pcd.point.positions = o3d.core.Tensor(xyz, o3d.core.float32, o3d.core.Device("CPU:0"))

        print(df.shape)
        if mode == "intensity":
            # Extract intensity values
            df = pd.read_csv(path_to_scene, skiprows=0, usecols=[0, 1, 2, 3])
            intensity = df.iloc[:, 3].to_numpy() / 1000
            # Set the colors of the point cloud using the intensity-based color map
            pcd.point.colors = o3d.core.Tensor(
                intensity, o3d.core.float32, o3d.core.Device("CPU:0"))
        elif mode == "x":
            # Extract intensity values
            x = df.iloc[:, 0].to_numpy() / 1000
            # Set the colors of the point cloud using the intensity-based color map
            pcd.point.colors = o3d.core.Tensor(
                x, o3d.core.float32, o3d.core.Device("CPU:0"))

        return pcd


def obtain_scenes(path_to_scenes, mode):
    print("Obtaining scenes from the path to the .txt files...")
    path_to_scenes_ext = os.path.join(path_to_scenes, "*.txt")

    filenames = [
        os.path.basename(abs_path) for abs_path in glob.glob(path_to_scenes_ext)
    ]
    # print(filenames)

    # The resolution
    res = np.float32(float(os.path.splitext((filenames[0].split("_")[-1]))[0]))

    # For offsetting frame indexing in case if we are working with padded output
    # Output should usually be padded anyways
    # Offset is the first frame, this line gets the minimum frame number
    offset = int(min(filenames, key=lambda x: int(
        (x.split("_"))[1])).split("_")[1])

    # Create our opener object (for inputs/outputs to be serializable)
    opener = PointCloudOpener()
    # Define the arguments that will be ran upon in parallel.
    args = [(frame + offset, res) for frame in range(len(filenames))]
    cores = 10

    from joblib import Parallel, delayed

    pcds = Parallel(n_jobs=cores)(  # Switched to loky backend to maybe suppress errors?
        delayed(opener.open_point_cloud)(path_to_scenes, frame, res, mode)
        for frame, res in tqdm(
            args,
            total=len(filenames),
            desc=f"Reading scenes to memory in parallel, using {cores} processes",
        )
    )

    return pcds

# Play our scenes using Open3D's visualizer
def visualize_replay(
    path_to_scenes: str,
    scenes_list: np.ndarray,
    vehicle_speed: np.float32 = 100,
    point_density: np.float32 = 1.0,
    mode: str = "default"
) -> str or int:
    # Obtain screen parameters for our video
    root = Tk()
    root.withdraw()
    SCREEN_WIDTH, SCREEN_HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()

    # Helper function to visualize the replay of our frames in a video format
    def replay_capture_frames():
        print(f"Visualizing the scenes given by path {path_to_scenes}")

        # creates an empty point cloud on which we will display stuff using the visualizer
        geometry = o3d.geometry.PointCloud()
        vis.add_geometry(geometry)

        # Obtain view control of the visualizer to change POV on setup
        # NOTE Currently, as of 5/19/2023, the get_view_control() method for the open3d.Visualizer class
        # only returns a copy of the view control as opposed to a reference.
        if o3d.__version__ == "0.17.0":
            print("Use the correct open3d version")
            sys.exit(1)
        else:
            ctr = vis.get_view_control()

        images_dir = os.path.join(os.getcwd(), "frame_images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        for frame, scene in enumerate(
            tqdm(scenes_list, desc="Replaying and capturing scenes")
        ):
            if mode == "default":
                geometry.points = scene.to_legacy().points  # IF THE SCENE IS IN TENSOR
            else:
                # For intensity colors
                xyz = scene.point.positions.numpy()  # IF THE SCENE IS IN TENSOR
                geometry.points = o3d.utility.Vector3dVector(xyz)
                scalar_val = scene.point.colors.numpy()
                normalizer = matplotlib.colors.Normalize(
                    np.min(scalar_val), np.max(scalar_val))
                las_rgb = matplotlib.cm.gray(normalizer(scalar_val))[:, :-1]
                geometry.colors = o3d.utility.Vector3dVector(las_rgb)

            if frame == 0:
                vis.add_geometry(geometry, reset_bounding_box=True)
            else:
                vis.update_geometry(geometry)

            # Set view of the live action Open3D replay
            if o3d.__version__ == "0.17.0":  # This probably doesn't work
                print("Use the correct open3d version")
                sys.exit(1)
            else:
                # Isometric front view
                if VIEW == "front":
                    ctr.set_front([-1, 0, 0])  
                    ctr.set_up([0, 0, 1])
                    ctr.set_lookat([18.5, 0, 1.8])
                    ctr.set_zoom(0.025)   
                elif VIEW == "isometric":
                    ctr.set_front([-1, -1, 1])  
                    ctr.set_up([0, 0, 1])
                    ctr.set_lookat([0, 0, 1.8])
                    ctr.set_zoom(ZOOM)

            """ Settings for POV of driver:
        ctr.set_front([-1, 0, 0])  
        ctr.set_up([0, 0, 1])
        ctr.set_lookat([18.5, 0, 1.8])
        ctr.set_zoom(0.025)    
      """
            """ Settings for isometric forward POV:
        ctr.change_field_of_view(step=50)
        ctr.set_front([-1, -1, 1])  
        ctr.set_up([0, 0, 1])
        ctr.set_lookat([0, 0, 1.8])
        ctr.set_zoom(0.3)  
      """

            # Update the renderer
            vis.poll_events()
            vis.update_renderer()

            img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            img = (img[:, :] * 255).astype(np.uint8)  # Normalize RGB to 8-bit

            # Capture the rendered point cloud to an RGB image for video output
            # frames.append(np.asarray(vis.capture_screen_float_buffer(do_render=True)))
            cv2.imwrite(filename=os.path.join(
                images_dir, f"{frame}.png"), img=img)

            # Play the scenes as it appears in the vehicle's speed
            # time.sleep((1*point_density)/(vehicle_speed/3.6))

        return images_dir, SCREEN_WIDTH, SCREEN_HEIGHT

    # Example taken from open3d non-blocking visualization...
    vis = o3d.visualization.Visualizer()

    while True:
        usr_inpt = input(
            f"Press 'p' to replay {len(scenes_list)} scenes given by {path_to_scenes} (press 'q' to exit), c to create video without replay (only works if you have saved images already): "
        )
        if usr_inpt == "p":
            vis.create_window(
                window_name=f"Scenes of {path_to_scenes}",
                width=SCREEN_WIDTH,
                height=SCREEN_HEIGHT,
                left=10,
                top=10,
                visible=True,
            )

            vis.set_full_screen(True)

            # View control options (also must be created befoe we can replay our frames)
            # Render options (must be c>reated before we can replay our frames)
            render_opt = vis.get_render_option()
            render_opt.point_size = 1.0
            render_opt.show_coordinate_frame = True
            render_opt.background_color = np.array(
                [16 / 255, 16 / 255, 16 / 255]
            )  # 8-bit RGB, (16, 16, 16)

            vis.poll_events()
            vis.update_renderer()

            frames, sw, sh = replay_capture_frames()
            return frames, sw, sh
        elif usr_inpt == "q":
            return
        elif usr_inpt == "c":
            images_dir = os.path.join(os.getcwd(), "frame_images")
            if not os.path.exists(images_dir):
                print("you will need to play the video to save the images")
                continue
            return images_dir, SCREEN_WIDTH, SCREEN_HEIGHT

    print("Visualization complete.")
    vis.clear_geometries()
    vis.destroy_window()

    return

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
        
        parser.add_argument("--input", type=str, default=None, help="Path to the .las file")

        parser.add_argument("--mode", type=str, default="default", help="Mode for the colors of the points", choices=["default", "intensity", "x"])
        
        parser.add_argument("--view", type=str, default="isometric", help="Option for the view of the points", choices=["front", "isometric"])

        parser.add_argument("--zoom", type=float, default=0.3125, help="Zoom level of car pov")

        return parser.parse_args()
    
    global ZOOM, VIEW
    args = parse_cmdline_args()
    args.scenes = os.environ["SCENES"]

    path_to_scenes = file_tools.obtain_scene_path()
    car_path = os.path.join(os.getcwd(), "frame_images/")

    if os.path.exists(car_path):
        shutil.rmtree(car_path)
        os.makedirs(car_path)
    ZOOM = args.zoom if args.zoom is not None else ZOOM
    VIEW = args.view if args.view is not None else VIEW
    # creating the video from the pov of the driver
    # this will prompt you to click p to play, click p otherwise it won't save your video

    print("Converting the .txt scenes to point clouds...")
    scenes = obtain_scenes(path_to_scenes, mode=args.mode)

    print("Creating video from car POV, please click 'p' when prompted...")
    frames, sw, sh = visualize_replay(path_to_scenes, scenes, mode=args.mode)

if __name__ == "__main__":
    main()