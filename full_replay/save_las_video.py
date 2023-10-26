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
from multiprocessing import Process, Lock, Queue

import shutil
from dotenv import load_dotenv

VIDEO_SPEED = 1


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
    fps = VIDEO_SPEED*np.ceil((vehicle_speed/3.6)/(1*point_density))
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
    for frame_i, filename in enumerate(tqdm(filenames, total=len(filenames), desc=f"Writing to video")):

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

        # source_text = f"Source: {os.path.basename(os.path.normpath(path_to_scenes))}"
        # # Get width and height of the source text
        # source_text_size = cv2.getTextSize(
        #     source_text, fontFace=font, fontScale=font_scale, thickness=font_thickness)[0]
        # source_xy = (w-source_text_size[0]-20, source_text_size[1]+20)
        # frame_annotated = annotate_frame(
        #     frame_annotated, source_text, source_xy)

        writer.write(cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB))

    # All done
    writer.release()
    print(f"Video replay has been written to {output_path}")
    return

def get_image_files(paths):
    # Read our frames from our temporary directory
    car_path, las_path = paths
    car_path_ext = os.path.join(car_path, '*.png')
    las_path_ext = os.path.join(las_path, '*.png')

    # Get list of filenames within our temporary directory
    car_images = [os.path.basename(abs_path)
                  for abs_path in glob.glob(car_path_ext)]
    car_images = sorted(car_images, key=lambda f: int(os.path.splitext(f)[0]))

    # Get list of filenames within our temporary directory
    las_images = [os.path.basename(abs_path)
                     for abs_path in glob.glob(las_path_ext)]
    las_images = sorted(
        las_images, key=lambda f: int(os.path.splitext(f)[0]))
    
    return car_images, las_images

def combine_images(images: tuple, paths: tuple, lIdx: int, rIdx: int, q: Queue, lock: Lock):
    out_path = os.path.join(os.getcwd(), "combined_images")
    
    car_images, las_images = images
    car_path, las_path = paths

    for i in tqdm(range(lIdx, rIdx), desc="Combining images"):
        car_image = os.path.join(car_path, car_images[i])
        las_image = os.path.join(las_path, las_images[i])

        img1 = cv2.imread(las_image)
        img2 = cv2.imread(car_image)

        (h1, w1) = img1.shape[:2]
        (h2, w2) = img2.shape[:2]
        
        out = np.zeros((h1, w1 + w2, 3), dtype="uint8")
        out[:, :w1] = img1
        out[:, w1:w1+w2] = img2

        cv2.imwrite(os.path.join(
            os.getcwd(), "combined_images", f"{i}.png"), out)

    # return the height and width to pass on to the create_video function

    lock.acquire()
    q.put((100 + h1 + h2, 100 + w1))
    lock.release()


def main():
    # Parse our command line arguments
    def parse_cmdline_args() -> argparse.Namespace:
        # use argparse to parse arguments from the command line
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--scenes", type=str, default=None, help="Path to the Vista output folder")

        parser.add_argument("--speed", type=float, default=4, help="Speed of video")

        parser.add_argument("--video_name", type=str, default="combined", help="Name of video file")


        return parser.parse_args()

    args = parse_cmdline_args()
    load_dotenv()
    args.scenes = os.environ["SCENES"] if args.scenes is None else args.scenes
    
    combined_path = os.path.join(os.getcwd(), "combined_images")
    if not os.path.exists(combined_path):
        os.makedirs(combined_path)
    else:
        shutil.rmtree(combined_path)
        os.makedirs(combined_path)
    global VIDEO_SPEED
    VIDEO_SPEED = args.speed
    name = args.video_name

    path_to_scenes = file_tools.obtain_scene_path(args)
    car_path = os.path.join(os.getcwd(), "frame_images/")
    las_path = os.path.join(os.getcwd(), "las_pov/")
    # combine images and create the video
    paths = (car_path, las_path)
    images = get_image_files(paths)

    # we'll have 10 processes

    interval_length = len(images[0])//10
    intervals = []
    for i in range(9):
        intervals.append((i*interval_length, (i+1)*interval_length))
    intervals.append((9*interval_length, len(images[0])))
    
    processes = []
    lock = Lock()
    q = Queue()
    for interval in intervals:
        p = Process(target=combine_images, args=(images, paths, interval[0], interval[1], q, lock))
        processes.append(p)

    print(f"Each running process is printing a progress bar, so it won't look consistent,\n just a general idea of how long it will take.")
    print(f"Combining {len(images[0])} images...")
    for p in processes:
        p.start()
    
    for p in processes:
        p.join()

    h, w = q.get()
    print(h, w)
    images_dir = os.path.join(os.getcwd(), "combined_images")
    create_video(images_dir, w, h, path_to_scenes, filename=f"{name}.mp4")


if __name__ == "__main__":
    main()
