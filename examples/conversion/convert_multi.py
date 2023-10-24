import h5py
import argparse
import laspy
import numpy as np
import zipfile
import math
import os
import time
import torch
from statistics import mean
import pandas as pd


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
    print(device)

    erase = False
    if args.input.endswith(".zip"):
        if (
            os.path.exists(f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}")
            == False
        ):
            print(
                f"Un-zipping to /tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}"
            )
            with zipfile.ZipFile(args.input, "r") as zip_ref:
                zip_ref.extractall("/tmp/lidar")
            print("Un-zipping done")
        else:
            erase = True

        args.input = f"/tmp/lidar/{args.input.split('/')[-1].split('.zip')[0]}"

    read_las = False
    while read_las == False:
        try:
            las = laspy.read(args.input)
            read_las = True
        except ValueError:
            print("Un-zipping not finished. Wait 5 seconds.")
            time.sleep(5)
        except OSError:
            exit(1)

    if erase and args.input.startswith("/tmp/lidar/") and args.input.endswith(".las"):
        os.remove(args.input)
        print(f"Removing {args.input}")

    trajectory = np.sort(
        las.points[
            np.array(
                np.where(
                    (las._points.scan_angle_rank == 0)
                    & (las._points.point_source_id == 2)
                )
            )
        ].array,
        order="gps_time",
    )[0]

    if args.frame:
        frame = args.frame
    else:
        frame = len(trajectory) // 2

    with open(f"./examples/vista_traces/lidar_{args.process}/log.txt", "a") as f:
        print(f"{args.input.split('/')[-1]} / frame # {frame}", file=f)

    try:
        pov = trajectory[frame]
    except IndexError:
        exit(1)

    last = trajectory[-1]
    if (
        (pov["X"] - last["X"]) ** 2
        + (pov["Y"] - last["Y"]) ** 2
        + (pov["Z"] - last["Z"]) ** 2
    ) ** (1 / 2) < 250000:
        exit(1)

    pov_X = pov["X"]
    pov_Y = pov["Y"]
    pov_Z = pov["Z"]
    pov_X_delta = []
    pov_Y_delta = []
    pov_Z_delta = []

    for i in range(1, 30):
        pov_next = trajectory[frame + i]
        pov_X_delta.append(pov_next["X"] - pov_X)
        pov_Y_delta.append(pov_next["Y"] - pov_Y)
        pov_Z_delta.append(pov_next["Z"] - pov_Z)

    pov_X_delta = mean(pov_X_delta)
    pov_Y_delta = mean(pov_Y_delta)
    pov_Z_delta = mean(pov_Z_delta)

    x = np.array(las.X)
    y = np.array(las.Y)
    z = np.array(las.Z)

    xyz = np.asarray([x, y, z], dtype=np.float64).T

    # Traslantion
    xyz -= np.array([pov_X, pov_Y, pov_Z])
    # Sensor at 1.2 meter above
    xyz[:, 2] -= 1200
    
    xyz_distance = np.sqrt(
        np.square(xyz[:, 0]) + np.square(xyz[:, 1]) + np.square(xyz[:, 2])
    )

    indices = np.where((xyz_distance < 245000))
    xyz = torch.tensor(xyz[indices]).to(device)

    # Rotation 1
    cos_1 = pov_X_delta / ((pov_X_delta**2 + pov_Y_delta**2) ** (0.5))
    sin_1 = pov_Y_delta / ((pov_X_delta**2 + pov_Y_delta**2) ** (0.5))
    xyz = torch.matmul(
        torch.tensor([[cos_1, sin_1, 0], [-sin_1, cos_1, 0], [0, 0, 1]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    # Rotation 2
    cos_2 = ((pov_X_delta**2 + pov_Y_delta**2) ** (0.5)) / (
        ((pov_X_delta**2 + pov_Y_delta**2) + pov_Z_delta**2) ** (0.5)
    )
    sin_2 = pov_Z_delta / (
        ((pov_X_delta**2 + pov_Y_delta**2) + pov_Z_delta**2) ** (0.5)
    )
    xyz = torch.matmul(
        torch.tensor([[cos_2, 0, -sin_2], [0, 1, 0], [sin_2, 0, cos_2]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    # Cross section angle
    cross_section = xyz[
        (xyz[:, 0] < 100)
        & (xyz[:, 0] > -100)
        & (xyz[:, 1] < 1000)
        & (xyz[:, 1] > -1000)
    ]

    tan_li = cross_section[:, 2] / cross_section[:, 1]

    tangent = torch.nanmean(tan_li)
    cross_angle = math.atan(tangent)
    cos_3 = math.cos(cross_angle)
    sin_3 = math.sin(cross_angle)

    xyz = torch.matmul(
        torch.tensor([[1, 0, 0], [0, cos_3, -sin_3], [0, sin_3, cos_3]])
        .double()
        .to(device),
        xyz.double().T,
    ).T

    xyz = xyz.cpu().numpy()

    with h5py.File(
        f"./examples/vista_traces/lidar_{args.process}/lidar_3d.h5", "w"
    ) as f:
        f["timestamp"] = [[0], [0.1], [0.2]]
        f["xyz"] = [xyz]
        f["intensity"] = [las.intensity[indices[0]]]

    f2 = h5py.File(f"./examples/vista_traces/lidar_{args.process}/lidar_3d.h5", "r")
    print(f2["timestamp"])
    print(f2["xyz"])
    print(f2["intensity"])

    pd.DataFrame(xyz).to_csv(
        f"./examples/vista_traces/lidar_{args.process}/lidar_3d.csv", index=False
    )


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to .las file to convert to .h5")
    parser.add_argument("--frame", type=int, help="Frame number")
    parser.add_argument("--process", type=int, help="Process number")
    args = parser.parse_args()

    main(args)
