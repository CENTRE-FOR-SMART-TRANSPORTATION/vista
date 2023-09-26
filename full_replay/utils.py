import open3d as o3d
import numpy as np
from classes import LasPointCloud, Trajectory, SensorConfig

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
