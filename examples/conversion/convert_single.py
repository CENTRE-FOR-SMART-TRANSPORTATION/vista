import h5py
import argparse
import laspy
import numpy as np
import zipfile
import math
import os
import torch
import pandas as pd

import gen_traj
from LasPointCloud import LasPointCloud


def main(args):
    # Use this if you want to manually generate a trajectory for each process
    # It may be faster to use a pregenerated trajectory
    usePregenerated = True

    device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    print(device)

    las = laspy.read(args.input)
    # las_offsets = las.header.offsets
    
    try:
        run_occlusions = args.occlusion
    except AttributeError:
        run_occlusions = False

    filename_cut = os.path.splitext(os.path.basename(args.input))[
        0
    ]  # Converted outputs are for a specific road section

    # Manually input pregenerated trajectories (for now)
    if usePregenerated:
        path_road_points = os.path.join(
            "./examples/Trajectory", filename_cut, "road_points.csv"
        )
        road_points = pd.read_csv(path_road_points, sep=",", header=None).values

        path_upwards = os.path.join(
            "./examples/Trajectory", filename_cut, "upwards.csv"
        )
        upwards = pd.read_csv(path_upwards, sep=",", header=None).values

        path_forwards = os.path.join(
            "./examples/Trajectory", filename_cut, "forwards.csv"
        )
        forwards = pd.read_csv(path_forwards, sep=",", header=None).values

        path_leftwards = os.path.join(
            "./examples/Trajectory", filename_cut, "leftwards.csv"
        )
        leftwards = pd.read_csv(path_leftwards, sep=",", header=None).values
    else:
        # Generate our trajectories as we go
        las_struct = LasPointCloud(
            las.x, las.y, las.z, las.gps_time, las.scan_angle_rank, las.point_source_id
        )

        traj = gen_traj.TrajectoryConfig(2.0, 1.0, 1.8)

        road_points, forwards, leftwards, upwards = gen_traj.generate_trajectory(
            True, las_struct, traj
        )

    trajectory = road_points
    i = args.frame
    print(f"Frame #: {i}")

    # Fix the z component of the forwards vector given that the other two are orthogonal
    forwards[i][2] = (
        -(upwards[i][0] * forwards[i][0] + upwards[i][1] * forwards[i][1])
        / upwards[i][2]
    )
    magnitude = (forwards[i][0] ** 2 + forwards[i][1] ** 2 + forwards[i][2] ** 2) ** (
        1 / 2
    )
    forwards /= magnitude

    # Local coordinates in mm
    pov_X = (trajectory[i][0]) * 1000
    pov_Y = (trajectory[i][1]) * 1000
    pov_Z = (trajectory[i][2]) * 1000

    # Global coordinates in mm
    x = (np.array(las.x)) * 1000
    y = (np.array(las.y)) * 1000
    z = np.array(las.z) * 1000
    xyz = np.asarray([x, y, z], dtype=np.float64).T

    # Traslantion
    xyz -= np.array([pov_X, pov_Y, pov_Z])  # Inverse: Add the pov.

    xyz_distance = np.sqrt(
        np.square(xyz[:, 0]) + np.square(xyz[:, 1]) + np.square(xyz[:, 2])
    )

    # mm distance less than the FOV
    indices = np.where((xyz_distance < args.range * 1000))
    xyz = torch.tensor(xyz[indices]).to(device)



    # Rotation 1 (yaw; about the z-axis, negative RHR)
    cos_1 = forwards[i][0] / ((forwards[i][0] ** 2 + forwards[i][1] ** 2) ** (0.5))
    sin_1 = forwards[i][1] / ((forwards[i][0] ** 2 + forwards[i][1] ** 2) ** (0.5))
    
    xyz = torch.matmul(
        torch.tensor([[cos_1, sin_1, 0], [-sin_1, cos_1, 0], [0, 0, 1]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    # Rotation 2 (pitch; about the y-axis, negative RHR)
    cos_2 = ((forwards[i][0] ** 2 + forwards[i][1] ** 2) ** (0.5)) / (
        ((forwards[i][0] ** 2 + forwards[i][1] ** 2) + forwards[i][2] ** 2) ** (0.5)
    )
    sin_2 = forwards[i][2] / (
        ((forwards[i][0] ** 2 + forwards[i][1] ** 2) + forwards[i][2] ** 2) ** (0.5)
    )
    
    xyz = torch.matmul(
        torch.tensor([[cos_2, 0, -sin_2], [0, 1, 0], [sin_2, 0, cos_2]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    # Rotation 3 (roll; about the x-axis; positive RHR)
    tangent = leftwards[i][2] / ((leftwards[i][0] ** 2 + leftwards[i][1] ** 2) ** (0.5))
    cross_angle = math.atan(tangent)
    cos_3 = math.cos(cross_angle)
    sin_3 = math.sin(cross_angle)

    xyz = torch.matmul(
        torch.tensor([[1, 0, 0], [0, cos_3, -sin_3], [0, sin_3, cos_3]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    # Translate by observer height (in mm)
    xyz[:, 2] -= 1800
    
    # Debug: Write segmented point cloud in global coordinates

    
    # Cull points by angle of sensor if we are not running occlusion
    if not run_occlusions:
        yaw_min = args.yaw_min
        yaw_max = args.yaw_max
        pitch_min = args.pitch_min
        pitch_max = args.pitch_max
        
        yaws = torch.atan2(xyz[:,1], xyz[:,0])*(180/np.pi)
        pitches = torch.atan(xyz[:,2]/(torch.linalg.norm(xyz[:,0:2], dim=1)))*(180/np.pi)
    
        yaw_mask = (yaws >= yaw_min) & (yaws <= yaw_max)
        pitch_mask = (pitches >= pitch_min) & (pitches <= pitch_max)
        filtered_mask = yaw_mask & pitch_mask
        
        xyz = xyz[filtered_mask]
        xyz /= 1000 # Convert back to mm for less file size

    xyz = xyz.cpu().numpy() 
    
    #pd.DataFrame(xyz).to_csv(
    #    f"./examples/vista_traces/lidar_3d.csv", header=False, index=False
    #)
    
    # Write unoccluded output
    if not run_occlusions:
        output_folder = f"./examples/vista_traces/lidar_output/{os.path.splitext(args.filename)[0]}_unoccluded/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_path = os.path.join(output_folder, f"output_{i}.txt")
        
        output_path = "".join(output_path.split(" "))

        pd.DataFrame(xyz).to_csv(
            output_path, header=False, index=False
        ) 
       
    else:
        if not os.path.exists(f"./examples/vista_traces/lidar_{args.process}"):
          os.makedirs(f"./examples/vista_traces/lidar_{args.process}")
        
        # Write intermediate file for Vista simulation
        with h5py.File(
            f"./examples/vista_traces/lidar_{args.process}/lidar_3d.h5", "w"
        ) as f:
            f["timestamp"] = [[0], [0.1], [0.2]]
            f["xyz"] = [xyz]
            f["intensity"] = [las.intensity[indices]]

        # Verify if intermediate file is being written
        f2 = h5py.File(f"./examples/vista_traces/lidar_{args.process}/lidar_3d.h5", "r")
        print(f2["timestamp"])
        print(f2["xyz"])
        print(f2["intensity"])


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input", type=str, help="Path to .las file to convert to .h5")
    parser.add_argument("--frame", type=int, help="Frame number")
    parser.add_argument("--range", type=int, help="Range distance in metres")
    parser.add_argument("--process", type=int, help="Process number")
    parser.add_argument("--occlusion", action="store_true", help="option to include occlusion or not")
    parser.add_argument("--filename", type=str, help="Filename of the las file")
    
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

    args = parser.parse_args()

    main(args)
