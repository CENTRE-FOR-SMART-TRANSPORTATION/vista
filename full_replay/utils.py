import open3d as o3d
import numpy as np
from classes import LasPointCloud, Trajectory, SensorConfig, PointCloudOpener
import os
import laspy
import json

import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import tkinter as tk
import sys
import glob
import time
import pickle

from tkinter import Tk
from pathlib import Path
from tqdm import tqdm
import utils

import matplotlib

# Open our .las point cloud into memory


def open_las(path: str) -> LasPointCloud:
    """
    Opens a .las file when prompted to do so. Can force a predetermined filename
    (default called as None for manual input)

    Arguments:
    verbose (bool): Setting to print extra information to the command line.

    predetermined_filename (string): The predetermined file name of the point cloud.
    User can be manually prompted to enter the point cloud, or it can be set to some
    point cloud via command line for automation. See main() for command line syntax.
    """
    las_filename = path

    # Obtain the las file name itself rather than the path for csv output
    las_filename_cut = os.path.basename(las_filename)

    # Note: lowercase dimensions with laspy give the scaled value
    raw_las = laspy.read(las_filename)
    las = LasPointCloud(
        raw_las.x,
        raw_las.y,
        raw_las.z,
        raw_las.gps_time,
        raw_las.scan_angle_rank,
        raw_las.point_source_id,
        raw_las.intensity,
        las_filename_cut,
    )

    return las


def las2o3d_pcd(las: LasPointCloud) -> o3d.geometry.PointCloud:
    """Reads the road section into memory such that we can visualize
    the sensor FOV as it goes through the road section.

    Args:
        las (file_tools.LasPointCloud): Our .las point cloud to read values from

    Returns:
        pcd (o3d.geometry.PointCloud): Our point cloud, replay compatible.

        filename (str): The respective filename of the las file.
    """

    pcd = o3d.geometry.PointCloud()

    las_xyz = np.vstack((las.getX(), las.getY(), las.getZ())).T
    pcd.points = o3d.utility.Vector3dVector(las_xyz)
    print("\nPoints loaded.")

    import matplotlib

    las_intensity = las.getIntensity()

    # Normalize intensity values to [0, 1], then assign RGB values
    normalizer = matplotlib.colors.Normalize(
        np.min(las_intensity), np.max(las_intensity))
    las_rgb = matplotlib.cm.gray(normalizer(las_intensity))[:, :-1]
    # cmap(las_intensity) returns RGBA, cut alpha channel
    pcd.colors = o3d.utility.Vector3dVector(las_rgb)
    print("Intensity colors loaded.")

    filename = las.getLasFileName()

    return pcd, filename


# Obtain the trajectory
def obtain_trajectory_details(path: str) -> Trajectory:
    """Obtains a pregenerated trajectory and reads each of them into
    a container class.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Trajectory: Container class for our imported trajectory data.
    """
    trajectory_folderpath = path

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

    # Read each of the csv files as numpy arrays
    trajectory_data = dict()

    for csv in trajectory_files:
        csv_noext = os.path.splitext(csv)[0]
        path_to_csv = os.path.join(trajectory_folderpath, csv)
        data = np.genfromtxt(path_to_csv, delimiter=",")
        trajectory_data[csv_noext] = data

    observer_points = trajectory_data["observer_points"]
    road_points = trajectory_data["road_points"]
    forwards = trajectory_data["forwards"]
    leftwards = trajectory_data["leftwards"]
    upwards = trajectory_data["upwards"]

    # Another sanity check
    assert (
        observer_points.shape
        == road_points.shape
        == forwards.shape
        == leftwards.shape
        == upwards.shape
    ), f"Bad trajectory files! One or more trajectories are missing points!"

    # Correct the z-component of our forward vector FIXME This is broken, fix later...
    useCorrectedZ = False
    if useCorrectedZ:
        print(f"Using the corrected z-compoment of the forward vector!")
        forwards[:, 2] = (
            -(upwards[:, 0] * forwards[:, 0] + upwards[:, 1] * forwards[:, 1])
            / upwards[:, 2]
        )

        magnitude = (
            forwards[:, 0] ** 2 + forwards[:, 1] ** 2 + forwards[:, 2] ** 2
        ) ** (1 / 2)

        forwards[:, 2] /= magnitude

    # Finally store the trajectory values into our object
    trajectory = Trajectory(
        observer_points=observer_points,
        road_points=road_points,
        forwards=forwards,
        leftwards=leftwards,
        upwards=upwards,
    )

    print(
        f"{road_points.shape[0]} trajectory points have been loaded for the corresponding trajectory folder {os.path.basename(trajectory_folderpath)}"
    )

    return trajectory


def open_sensor_config_file(path: str) -> SensorConfig:
    """Opens the sensor configuration file from command-line argument or
    through UI.

    Args:
        args (argparse.Namespace): Contains the command-line arguments.

    Returns:
        cfg (SensorConfig): Container class containing the sensor configuration
        parameters.
    """
    # tStart = perf_counter()
    sensorcon_filepath = path

    with open(sensorcon_filepath, 'r') as f:
        data = f.read()

    sensor_cfg_dict = json.loads(data)

    # Create container object
    cfg = SensorConfig(
        sensor_cfg_dict["numberSensors"],
        sensor_cfg_dict["horizAngRes"],
        sensor_cfg_dict["verticAngRes"],
        sensor_cfg_dict["e_low"],
        sensor_cfg_dict["e_high"],
        sensor_cfg_dict["a_low"],
        sensor_cfg_dict["a_high"],
        sensor_cfg_dict["r_low"],
        sensor_cfg_dict["r_high"]
    )

    cfg.sensor_config_filename = os.path.basename(sensorcon_filepath)

    # tStop = perf_counter()

    # print(f"Loading took {(tStop-tStart):.2f}s.")

    return cfg


# Converted from generate_sensor_points_cell.m
# TODO Expand the sensor point generation for multiple sensors...
def generate_sensor_points(sensor_config: SensorConfig) -> list:
    """Creates a set of XYZ points that represent the FOV of the sensor
    configuration in meters, for each sensor.

    Args:
        sensor_config (SensorConfig): Container class for
        the sensor configuration.

    Returns:
        points (list): List containing the XYZ points of the form 
        np.ndarray that make up the FOV of each sensor.
    """

    # print(f"\nGenerating FOV points for {sensor_config.getNumberSensors()} sensor(s)!")
    # tStart = perf_counter()

    # Override the resolution
    verticalRes = 2
    horizontalRes = 2

    # Gammas correspond to vertical angles
    total_gammas = np.int32(
        np.floor(
            np.abs(sensor_config.getEHigh() -
                   sensor_config.getELow())/verticalRes
        )
    )
    gammas = np.linspace(
        sensor_config.getEHigh(),
        sensor_config.getELow(),
        total_gammas,
    ).reshape(total_gammas, 1)

    # Now create the points that make up the surfaces of the sensor FOV
    points = []  # Container of points for each sensor
    # For now we will only care about one sensor
    # TODO Expand code for multi-sensor configurations (tesla_day.json)
    for i in range(sensor_config.getNumberSensors()):

        # For multisensor configurations in json files, each field is a list.
        # This shouldn't be too bad to expand for other sensors

        # Thetas correspond to horizontal angles
        total_thetas = np.int32(
            np.floor(
                np.abs(sensor_config.getAHigh() -
                       sensor_config.getALow())/horizontalRes
            )
        )
        thetas = np.linspace(
            -sensor_config.getALow(),
            -sensor_config.getAHigh(),
            total_thetas,
        ).reshape(1, total_thetas)

        # Obtain XYZ points that will make up the front of the FOV
        fronts = np.hstack(  # X Y Z
            (
                np.reshape(np.cos(np.deg2rad(gammas))*np.cos(np.deg2rad(thetas)) * \
                           sensor_config.getRHigh(), (total_gammas*total_thetas, 1), order='F'),
                np.reshape(np.cos(np.deg2rad(gammas))*np.sin(np.deg2rad(thetas)) * \
                           sensor_config.getRHigh(), (total_gammas*total_thetas, 1), order='F'),
                np.reshape(np.sin(np.deg2rad(gammas))*np.ones(
                    thetas.shape[1]) * sensor_config.getRHigh(), (total_gammas*total_thetas, 1), order='F')
            )
        )

        # Obtain ranges
        total_vert_ranges = np.int32(
            np.floor(
                1/np.sin(np.deg2rad(verticalRes))
            )
        )

        vert_ranges = np.linspace(
            0,
            sensor_config.getRHigh(),
            total_vert_ranges
        ).reshape(total_vert_ranges, 1)

        total_horz_ranges = np.int32(
            np.floor(
                1/np.sin(np.deg2rad(horizontalRes))
            )
        )

        horz_ranges = np.linspace(
            0,
            sensor_config.getRHigh(),
            total_horz_ranges
        ).reshape(1, total_horz_ranges)

        # i'm not even sure if this this right but we will have to see
        if ((sensor_config.getALow() != -180) or (sensor_config.getAHigh() != 180)):

            # Obtain XYZ points that will make up the left side of the FOV
            left_side = np.hstack(  # X Y Z
                (
                    np.reshape(np.cos(np.deg2rad(gammas))*horz_ranges*np.cos(
                        np.deg2rad(-sensor_config.getALow())), (total_gammas*total_horz_ranges, 1), order='F'),
                    np.reshape(np.cos(np.deg2rad(gammas))*horz_ranges*np.sin(
                        np.deg2rad(-sensor_config.getALow())), (total_gammas*total_horz_ranges, 1), order='F'),
                    np.reshape(np.sin(np.deg2rad(gammas))*horz_ranges,
                               (total_gammas*total_horz_ranges, 1), order='F')
                )
            )

            # Obtain XYZ points that will make up the right side of the FOV
            right_side = np.hstack(  # X Y Z
                (
                    np.reshape(np.cos(np.deg2rad(gammas))*horz_ranges*np.cos(
                        np.deg2rad(-sensor_config.getAHigh())), (total_gammas*total_horz_ranges, 1), order='F'),
                    np.reshape(np.cos(np.deg2rad(gammas))*horz_ranges*np.sin(
                        np.deg2rad(-sensor_config.getAHigh())), (total_gammas*total_horz_ranges, 1), order='F'),
                    np.reshape(np.sin(np.deg2rad(gammas))*horz_ranges,
                               (total_gammas*total_horz_ranges, 1), order='F')
                )
            )

        else:
            # Sensor FOV already covers from -180 to 180 degrees, we don't need left and right sides
            left_side = np.zeros((0, 3))
            right_side = left_side

        # i'm not even sure if this this right but we will have to see
        # Obtain XYZ points that will make up the top side of the FOV
        top_side = np.hstack(  # X Y Z
            (
                np.reshape(np.cos(np.deg2rad(sensor_config.getEHigh()))*vert_ranges*np.cos(
                    np.deg2rad(thetas)), (total_thetas*total_vert_ranges, 1), order='F'),
                np.reshape(np.cos(np.deg2rad(sensor_config.getEHigh()))*vert_ranges * \
                           np.sin(np.deg2rad(thetas)), (total_thetas*total_vert_ranges, 1), order='F'),
                np.reshape(np.sin(np.deg2rad(sensor_config.getEHigh()))*vert_ranges * \
                           np.ones(thetas.shape[1]),   (total_thetas*total_vert_ranges, 1), order='F')
            )
        )

        # i'm not even sure if this this right but we will have to see
        # Obtain XYZ points that will make up the bottom side of the FOV
        bot_side = np.hstack(  # X Y Z
            (
                np.reshape(np.cos(np.deg2rad(sensor_config.getELow()))*vert_ranges*np.cos(
                    np.deg2rad(thetas)), (total_thetas*total_vert_ranges, 1), order='F'),
                np.reshape(np.cos(np.deg2rad(sensor_config.getELow()))*vert_ranges * \
                           np.sin(np.deg2rad(thetas)), (total_thetas*total_vert_ranges, 1), order='F'),
                np.reshape(np.sin(np.deg2rad(sensor_config.getELow()))*vert_ranges * \
                           np.ones(thetas.shape[1]),   (total_thetas*total_vert_ranges, 1), order='F')
            )
        )

        # Now that we have all of the XYZ points that consist the FOV surfaces, we can
        # construct the sensor FOV as a set of 3D points.
        out = np.concatenate(
            (fronts, left_side, right_side, top_side, bot_side), axis=0)
        points.append(out)  # Multisensor configuration

        # tStop = perf_counter()
        # print(f"FOV point generation took {(tStop-tStart):.2f}s.")

    return points


# Converted from make_sensor_las_file.m
def align_sensor_points(fov_points: list, trajectory: Trajectory, observer_point: int) -> list or int:
    """Aligns sensor points to the vehicle's orientation and position 
    at the provided scene number. Output will be in global coordinates
    such that it can be easily superimposed onto the road section itself.

    Args:
        fov_points (list): List containing the XYZ points that make up the sensor
        FOV, for each sensor.
        trajectory (Trajectory): Container class containing the trajectory parameters.
        observer_point (int): Observer point detailing the scene at where
        FOV points should be translated

    Returns:
        transformed_points (list): List containing the XYZ points that make up the sensor
        FOV, for each sensor after our transformation.
    """

    # In case if the user does not input a flag for the observer point
    total_road_points = trajectory.getObserverPoints().shape[0]
    if observer_point == None:
        observer_point = int(
            input(f"Enter the observer point (from 0 to {total_road_points}): "))
    if ((observer_point > total_road_points) or (observer_point < 0)):
        raise ValueError("Observer point is out of range!")

    # print("\nAligning FOV points!")
    # tStart = perf_counter()

    # Rotation matrices are formed as this in the first two dimensions:
    # (note that this is a 2D matrix, the third dimension is the ith rotation matrix)
    # [ fx_i fy_i fz_i ]
    # [ lx_i ly_i lz_i ]
    # [ ux_i uy_i uz_i ]
    #
    # For a point given by [x y z] (row vector):
    # [x y z]*R will take our points from RELATIVE to GLOBAL coordiantes
    # [x y z]*R' (R transposed) will take our points from GLOBAL to RELATIVE coordinates

    # Obtain rotation matrices
    # Equivalent to the implementation in MATLAB
    rotation_matrices = np.reshape(
        np.hstack((trajectory.getForwards(),
                  trajectory.getLeftwards(), trajectory.getUpwards())),
        (trajectory.getObserverPoints().shape[0], 3, 3),
        order='F'
    )

    rotation_matrices = np.transpose(rotation_matrices, (2, 1, 0))

    # Now we will translate our FOV points to the observer point and align it with the trajectory
    # Also equivalent to the implementation in MATLAB
    transformed_points = []
    for sensorpoints in fov_points:
        out = (
            np.matmul(sensorpoints[(sensorpoints[:, 2] > -1.8), :],
                      rotation_matrices[:, :, observer_point])
            +
            trajectory.getObserverPoints()[observer_point, :]
        )
        transformed_points.append(out)

    # tStop = perf_counter()
    # print(f"FOV point alignment took {(tStop-tStart):.2f}s.")

    return transformed_points, observer_point

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

    pcds = Parallel(n_jobs=cores)(  # Switched to loky backend to maybe suppress errors?
        delayed(opener.open_point_cloud)(path_to_scenes, frame, res)
        for frame, res in tqdm(
            args,
            total=len(filenames),
            desc=f"Reading scenes to memory in parallel, using {cores} processes",
        )
    )

    return pcds
