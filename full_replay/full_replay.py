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

import tempfile

from tkinter import Tk
from pathlib import Path
import tqdm
import utils

import matplotlib
from classes import SensorConfig, Trajectory


class PointCloudOpener:
    # Opens one specified point cloud as a Open3D tensor point cloud for parallelism
    def open_point_cloud(
        self, path_to_scenes: str, frame: int, res: np.float32
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
        # print(f"Opening {scene_name} as pcd...")

        # Skip our header, and read only XYZ coordinates
        df = pd.read_csv(path_to_scene, skiprows=0, usecols=[0, 1, 2, 3])
        xyz = df.iloc[:, :3].to_numpy() / 1000  # Extract XYZ coordinates
        intensity = df.iloc[:, 3].to_numpy()      # Extract intensity values

        # Create Open3D point cloud object with tensor values.
        # For parallelization, outputs must be able to be serialized
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        normalizer = matplotlib.colors.Normalize(
            np.min(intensity), np.max(intensity))
        las_rgb = matplotlib.cm.gray(normalizer(intensity))[:, :-1]
        # cmap(las_intensity) returns RGBA, cut alpha channel
        pcd.colors = o3d.utility.Vector3dVector(las_rgb)

        return pcd


def obtain_scenes(path_to_scenes):
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

    pcds = Parallel(n_jobs=cores, backend="loky")(  # Switched to loky backend to maybe suppress errors?
        delayed(opener.open_point_cloud)(path_to_scenes, frame, res)
        for frame, res in tqdm.tqdm(
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
            tqdm.tqdm(scenes_list, desc="Replaying and capturing scenes")
        ):
            
            geometry.points = scene.points # IF THE SCENE IS IN TENSOR
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
                # ctr.change_field_of_view(step=50)
                ctr.set_front([-1, 0, 0])
                ctr.set_up([0, 0, 1])
                ctr.set_lookat([18.5, 0, 1.8])
                ctr.set_zoom(0.025)

            """ Settings for POV of driver:
        ctr.set_front([-1, 0, 0])  
        ctr.set_up([0, 0, 1])
        ctr.set_lookat([18.5, 0, 1.8])
        ctr.set_zoom(0.025)    
      """
            """ Settings for isometric forward POV:
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


def create_video(images_dir: str, w: int, h: int, path_to_scenes: str, vehicle_speed: np.float32 = 100, point_density: np.float32 = 1.0, filename: str = "") -> None:
    """Creates a video from the recorded frames.

    Args:
        frames (list): _description_
        w (int): _description_
        h (int): _description_
        path2scenes (str): _description_
        vehicle_speed (np.float32, optional): _description_. Defaults to 100.
        point_density (np.float32, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    # Helper function to annotate text onto a frame
    def annotate_frame(image: np.ndarray, text: str, coord: tuple) -> np.ndarray:
        font = cv2.FONT_HERSHEY_SIMPLEX
        annotated_image = cv2.putText(
            img=image,
            text=text,
            org=coord,
            fontFace=font,            # Defined below
            fontScale=font_scale,     #
            color=font_color,         #
            thickness=font_thickness,
            lineType=cv2.LINE_AA
        )

        return annotated_image

    # Get filename of the recorded visualization
    if filename == "":
        filename = f"{os.path.basename(os.path.normpath(path_to_scenes))[:-1]}.mp4"

    output_folder = os.path.join(os.getcwd(), "videos")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, filename)

    # Configure video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_speed = 2 # to increase or decrease the speed of the video
    fps = fps_speed*np.ceil((vehicle_speed/3.6)/(1*point_density))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Parameters for annotating text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 1
    font_color = (255, 255, 255)  # 8-bit RGB

    # Read our frames from our temporary directory
    path_to_images_ext = os.path.join(images_dir, '*.png')

    # Get list of filenames within our temporary directory
    filenames = [os.path.basename(abs_path)
                 for abs_path in glob.glob(path_to_images_ext)]
    filenames = sorted(filenames, key=lambda f: int(os.path.splitext(f)[0]))

    # Now we will create our video
    print("")
    for frame_i, filename in enumerate(tqdm.tqdm(filenames, total=len(filenames), desc=f"Writing to video")):

        img = cv2.imread(filename=os.path.join(images_dir, filename))

        # Open3D normalizes the RGB values from 0 to 1, while OpenCV
        # requires RGB values from 0 to 255 (8-bit RGB)
        # img = (img[:,:]*255).astype(np.uint8)

        # Annotate our image
        progress_text = f"Frame {str(frame_i+1).rjust(len(str(len(filenames))), '0')}/{len(filenames)}"
        # Get width and height of thes source text
        progress_text_size = cv2.getTextSize(
            progress_text, fontFace=font, fontScale=font_scale, thickness=font_thickness)[0]
        progress_xy = (20, progress_text_size[1]+20)
        frame_annotated = annotate_frame(img, progress_text, progress_xy)

        source_text = f"Source: {os.path.basename(os.path.normpath(path_to_scenes))}"
        # Get width and height of the source text
        source_text_size = cv2.getTextSize(
            source_text, fontFace=font, fontScale=font_scale, thickness=font_thickness)[0]
        source_xy = (w-source_text_size[0]-20, source_text_size[1]+20)
        frame_annotated = annotate_frame(
            frame_annotated, source_text, source_xy)

        writer.write(cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB))

    # All done
    writer.release()
    print(f"Video replay has been written to {output_path}")
    return

# FIXME Sometimes the animation does not play at all
# Replays the sensor FOV as our simulated vehicle goes down the road section itself.


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

    # Helper function to set the visualizer POV
    def set_visualizer_pov(mode: str) -> None:
        if mode == 'center':
            centerpt = traj.getRoadPoints()[np.floor(
                num_points/2).astype(int), :].T.reshape((3, 1))
            # Set the view at the center, top down
            ctr.set_front([-1, 0, 1])
            ctr.set_up([0, 0, 1])
            ctr.set_lookat(centerpt)
            ctr.set_zoom(0.3125)
        elif mode == "follow":
            # Follow the vehicle as it goes through the map
            ctr.set_front([0, 1, 1])
            ctr.set_up([0, 0, 1])
            # Center the view around the sensor FOV
            ctr.set_lookat(traj.getRoadPoints()[frame, :])
            ctr.set_zoom(0.3125)

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

    geometry = o3d.geometry.PointCloud()
    vis.add_geometry(geometry)

    vis.poll_events()
    vis.update_renderer()

    # Begin our replay of the sensor FOV
    num_points = traj.getNumPoints()
    fov_points = utils.generate_sensor_points(cfg)
    if not os.path.exists(sensor_images_path):
        os.makedirs(sensor_images_path)

    for frame in range(0+offset, num_points-offset):
        # Get sensor FOV
        aligned_fov_points, _ = utils.align_sensor_points(
            fov_points, traj, frame)

        # TODO Accomodate multi-sensor configurations
        # (first do this through sensorpoints.py, note that fov_points is a list of all the XYZ sensor points)
        geometry.points = o3d.utility.Vector3dVector(
            aligned_fov_points[0])  # [0] temp for single_sensor
        geometry.colors = o3d.utility.Vector3dVector(np.ones(
            (aligned_fov_points[0].shape[0], 3), dtype=np.float64))  # [0] temp for single_sensor

        # Set the view at the center
        set_visualizer_pov('center')

        # Then update the visualizer
        if frame == 0+offset:
            vis.add_geometry(geometry, reset_bounding_box=False)
        else:
            vis.update_geometry(geometry)

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

# main code that runs everything


path_to_scenes = os.path.abspath(os.environ["PATH_TO_SCENES"])

print(
    f"You have chosen the directory {path_to_scenes} as the path to the .txt files")

scenes_list_path = os.path.join(os.getcwd(), 'scenes_list.pkl')
scenes = None
if not os.path.exists(scenes_list_path):
    scenes = obtain_scenes(path_to_scenes)
    # print("in if", type(scenes))
    with open(scenes_list_path, "wb") as f:
        pickle.dump(scenes, f)
else:
    print("Loading saved list...")
    with open(scenes_list_path, "rb") as f:
        scenes = pickle.load(f)
        # print("in else", type(scenes))

print(type(scenes[0]))

# creating the video from the pov of the driver

frames, sw, sh = visualize_replay(path_to_scenes, scenes)
print(frames, sw, sh)
create_video(frames, sw, sh, path_to_scenes)


# creating the video from sensor fov

# traj_path = os.path.abspath(os.environ["TRAJ_PATH"])
# # traj = utils.obtain_trajectory_details(traj_path)

# cfg_path = os.path.abspath(os.environ["SENSOR_PATH"])
# # cfg = utils.open_sensor_config_file(cfg_path)

# las_path = os.path.abspath(os.environ["LAS_FILE_PATH"])
# road = utils.open_las(las_path)


# vista_output_path = os.path.abspath(os.environ["VISTA_OUTPUT_PATH"])

# car_path = os.path.join(os.getcwd(), "frame_images/")
# sensor_images_path = os.path.join(os.getcwd(), "fov/")
# graph_path = os.path.join(os.getcwd(), "plt_images/")


# screen_wh = obtain_screen_size()
# frame_offset = check_for_padded(path_to_scenes)

# road_o3d, src_name = utils.las2o3d_pcd(road)
# render_sensor_fov(cfg=cfg,
#                   traj=traj,
#                   road=road_o3d,
#                   src_name=src_name,
#                   sensor_images_path=sensor_images_path,
#                   screen_width=screen_wh[0],
#                   screen_height=screen_wh[1],
#                   offset=frame_offset
#                   )


# combine images

def combine_images(car_path: str, sensor_path: str, graph_path: str):
    out_path = os.path.join(os.getcwd(), "combined_images")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Read our frames from our temporary directory
    car_path_ext = os.path.join(car_path, '*.png')
    sensor_path_ext = os.path.join(sensor_path, '*.png')
    graph_path_ext = os.path.join(graph_path, "*.png")

    # Get list of filenames within our temporary directory
    car_images = [os.path.basename(abs_path)
                  for abs_path in glob.glob(car_path_ext)]
    car_images = sorted(car_images, key=lambda f: int(os.path.splitext(f)[0]))

    # Get list of filenames within our temporary directory
    sensor_images = [os.path.basename(abs_path)
                     for abs_path in glob.glob(sensor_path_ext)]
    sensor_images = sorted(
        sensor_images, key=lambda f: int(os.path.splitext(f)[0]))

    graph_images = [os.path.basename(abs_path)
                    for abs_path in glob.glob(graph_path_ext)]
    graph_images = sorted(
        graph_images, key=lambda f: int(os.path.splitext(f)[0]))

    for i in range(len(car_images)):
        car_image = os.path.join(car_path, car_images[i])
        sensor_image = os.path.join(sensor_path, sensor_images[i])
        graph_image = os.path.join(graph_path, graph_images[i])

        img1 = cv2.imread(sensor_image)
        img2 = cv2.imread(car_image)
        img3 = cv2.imread(graph_image)

        (h1, w1) = img1.shape[:2]
        (h2, w2) = img2.shape[:2]
        (h3, w3) = img3.shape[:2]

        img1 = img1[h1//5:h1-(h1//3 + h1//10), :]
        img2 = img2[h2//7:, :]
        
        (h1, w1) = img1.shape[:2]
        (h2, w2) = img2.shape[:2]


        scale_percent = 90  # percent of original size
        w3 = int(img3.shape[1] * scale_percent / 100)
        h3 = int(img3.shape[0] * scale_percent / 100)
        dim = (w3, h3)

        # resize image
        resized = cv2.resize(img3, dim, interpolation=cv2.INTER_AREA)

        w1_new = w2 - w3
        img1_scale_percent = int(w1_new * (100 / img1.shape[1]))
        h1_new = int(img1.shape[0] * img1_scale_percent / 100)
        dim = (w1_new, h1_new)
        img1_resized = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)

        out = np.zeros((h1 + h2, w1, 3), dtype="uint8")

        out[0:h1, 0:w1_new] = img1_resized
        out[h1:h1+h2, 0:w1] = img2
        out[0:h1, w1_new:w1_new+w3] = resized

        return h1+h2, w1

        cv2.imwrite(os.path.join(
            os.getcwd(), "combined_images", f"{i}.png"), out)

    # return the height and width to pass on to the create_video function
    return h1+h2, w1

# h, w = combine_images(car_path, sensor_images_path, graph_path)
# images_dir = os.path.join(os.getcwd(), "combined_images")
# create_video(images_dir, w, h, path_to_scenes, filename="combined.mp4")


# data rate stuff
