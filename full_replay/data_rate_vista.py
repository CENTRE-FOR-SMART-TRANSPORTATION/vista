import json
import os
import math
import re
import glob
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from mplcursors import cursor
import matplotlib.animation as animation
import pickle
import file_tools

USE_VOLUMETRIC = True
USE_CARTESIAN = True
x_res = 0.11
y_res = 0.11
z_res = 0.11

# gets the first number in a string (read left to right)
# used to get the frame number out of the filename
# assumes that the frame number is the first number in the filename


def get_first_number(substring: str):
    numbers = re.findall(r'\d+', substring)
    if numbers:
        return int(numbers[0])
    else:
        return None

# gets the substring from the end of a string till the first forward
# or backwards slash encountered


def get_folder(string: str):
    match = re.search(r"([\\/])([^\\/]+)$", string)
    if match:
        result = match.group(2)
        return result
    return None

# Called upon in the parallel for loop


def multiprocessed_vol_funct(input_tuple: tuple):
    """Perfroms the volumetric method for a single scene.
    Called in the parallelized for loop.

    Args:
        input_tuple (tuple): Our input tuple of the form
        (voxel_rsize, voxel_asize, voxel_esize, data, i, vistaoutput_path, point_density, max_volume), where:
         - voxel_rsize: Range precision in meters
         - voxel_asize: Azimuth precision in meters
         - voxel_esize: Elevation precision in meters
         - data: Our sensor configuration
         - point_density: Density of the points in points per meter
         - max_volume: The max possible volume for all the voxels

    Returns:
        i (int): The frame number
        output (list): The ratio of occupied volume to total volume.
    """
    voxel_rsize, voxel_asize, voxel_esize, data, i, vistaoutput_path, point_density, max_volume, global_offset = input_tuple
    filename = "output_" + str(i) + "_0.11.txt"
    f = os.path.join(vistaoutput_path, filename)
    file = np.loadtxt(f, delimiter=',', skiprows=1)
    numFrame = get_first_number(filename)
    pc = np.divide(file, 1000)  # convert from mm to meters
    inputExtended = (pc, voxel_rsize, voxel_asize, voxel_esize, data, i)

    occ = occupancy_volume(inputExtended)/max_volume

    return i+global_offset, [((numFrame+global_offset) * point_density), occ]

# Called upon in the parallel for loop


def multiprocessed_count_funct(input_tuple: tuple):
    """Perfroms the simple method for a single scene.
    Called in the parallelized for loop.

    Args:
        input_tuple (tuple): Our input tuple of the form
        (voxel_rsize, voxel_asize, voxel_esize, data, i, vistaoutput_path, point_density), where:
         - voxel_rsize: Range precision in meters
         - voxel_asize: Azimuth precision in meters
         - voxel_esize: Elevation precision in meters
         - data: Our sensor configuration
         - point_density: Density of the points in points per meter

    Returns:
        i (int): The frame number
        output (list): The ratio of occupied volume to total volume.
    """
    voxel_rsize, voxel_asize, voxel_esize, data, i, vistaoutput_path, point_density, total_voxels, global_offset = input_tuple
    filename = "output_" + str(i) + "_0.11.txt"
    f = os.path.join(vistaoutput_path, filename)
    try:
        file = np.loadtxt(f, delimiter=',', skiprows=1)
    except ValueError:
        print(f"ERROR AT FRAME {i}")
    numFrame = get_first_number(filename)
    pc = np.divide(file, 1000)  # convert from mm to meters
    inputExtended = (pc, voxel_rsize, voxel_asize,
                     voxel_esize, data, i, total_voxels)

    return i+global_offset, [((numFrame+global_offset) * point_density), occupancy_count(inputExtended)]

# https://github.com/numpy/numpy/issues/5228
# http://matlab.izmiran.ru/help/techdoc/ref/cart2sph.html#:~:text=cart2sph%20(MATLAB%20Functions)&text=Description-,%5BTHETA%2CPHI%2CR%5D%20%3D%20cart2sph(X%2C,and%20Z%20into%20spherical%20coordinates.


def cart2sph(x, y, z) -> np.ndarray:
    # Converted from matlab's cart2sph() function.
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

# Helper function to calculate the total volume of occupied voxels in a particular scene
# input_tuple: (pc,voxel_rsize, voxel_asize, voxel_esize, data, i)


def occupancy_volume(input: tuple) -> np.float32:
    """Computes the volume of all occupied voxels,
    with working spherical coordinates.

    Args:
        input (tuple): Tuple of form (pc, r_size, azimuth_size, elevation_size, sensorcon).
         - pc: The point cloud to voxelize.
         - r_size: Voxel size of the spherical coordinate; radius of sphere.
         - azimuth_size: Azimuth angle of each voxel from the center
         - elevation_size: Elevation angle of each voxel from the center
         - sensorcon: Our sensor configuration.

    Returns:
        total_volume (np.float32): The volume of all of occupied voxels in the scene.
    """

    # Working with the Vista output point cloud

    pc = input[0]
    r_size = input[1]
    azimuth_size = input[2]
    elevation_size = input[3]
    sensorcon = input[4]

    # Filter out the x y z components
    xyz_point_cloud_data = pc[:, 0:3]

    if not USE_CARTESIAN:
        # Converting the point cloud into spherical coordinates
        azimuth, elevation, r = cart2sph(
            xyz_point_cloud_data[:, 0], xyz_point_cloud_data[:, 1], xyz_point_cloud_data[:, 2])

        # Converting from radians to Degrees
        # For both azimuth and elevation, both the min are now zero, and the max
        # has been changed from 180 to 360 for azimuth, and from 90 to 180 for
        # elevation. We basically got rid of negative angle and shifted the
        # notation by the respectable amount.
        azimuth = np.rad2deg(azimuth) + 180
        azimuth = np.mod(azimuth, 360)
        elevation = np.rad2deg(elevation) + 90
        elevation = np.mod(elevation, 180)
        spherical_point_cloud_data = np.transpose(
            np.array([r, elevation, azimuth]))

        # Origin coordinate setting. Where the sensor is. Move to outside the
        # function in the future.
        voxel_struct = {}
        voxel_struct["r_size"] = r_size
        voxel_struct["a_size"] = azimuth_size
        voxel_struct["e_size"] = elevation_size

        # Removing values that are over the constraints
        # Range
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 0] < sensorcon["r_high"]]
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 0] >= sensorcon["r_low"]]

        # Elevation range
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 1] < sensorcon["e_high"]]
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 1] >= sensorcon["e_low"]]
        # Azimuth range
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 2] < sensorcon["a_high"]]
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 2] >= sensorcon["a_low"]]

        # Spherical voxelization
        # Think of each voxel as a dV element in spherical coordinates, except that its
        # not of infinitesmal size. We are simply making an evenly spaced grid (in spherical coordinates)
        # of the solid that is bounded the sensor FOV.
        #
        # Here we take the real coordinates of each point, and convert to voxel indices
        # We take the floor of the voxel indices that are 'close enough' to each other;
        # i.e., duplicate indices correspond to multiple points within a voxel
        spherical_point_cloud_data[:, 2] = np.floor(
            spherical_point_cloud_data[:, 2]/voxel_struct["a_size"])
        spherical_point_cloud_data[:, 1] = np.floor(
            spherical_point_cloud_data[:, 1]/voxel_struct["e_size"])
        spherical_point_cloud_data[:, 0] = np.floor(
            spherical_point_cloud_data[:, 0]/voxel_struct["r_size"])

        # Only keep unique voxels
        # Removing duplicates creates the voxel data and also sorts it
        # We are also handling occlusions by sorting the original coordinates by
        # the distance from the sensor. Then we run unique on the azimuth and
        # elevation angle. The index output from that will be used to basically get
        # rid of all the voxels behind a certain angle coordinate.
        spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 0].argsort(
        )]
        vx, unique_indices = np.unique(
            spherical_point_cloud_data, axis=0, return_index=True)
        vx = vx[np.argsort(unique_indices)]

        # Finding volume
        # Simply take the integral of the all the voxels
        # Getting the range for integration
        a_low = vx[:, 2] * voxel_struct["a_size"]
        e_low = vx[:, 1] * voxel_struct["e_size"]
        r_low = vx[:, 0] * voxel_struct["r_size"]
        a_high = a_low + voxel_struct["a_size"]
        e_high = e_low + voxel_struct["e_size"]
        r_high = r_low + voxel_struct["r_size"]

        volume = (1/3)*((np.power(r_high, 3)) - (np.power(r_low, 3)))\
            * (np.cos(np.deg2rad(e_low)) - np.cos(np.deg2rad(e_high)))\
            * (np.deg2rad(a_high) - np.deg2rad(a_low))

        # Just sum the volume matrix
        total_volume = sum(volume)

    else:

        vx = np.zeros((xyz_point_cloud_data.shape[0], 4))

        # Fourth column is range which we will sort later on.
        vx[:, 3] = np.linalg.norm(xyz_point_cloud_data, axis=1)

        vx[:, 0] = np.floor(xyz_point_cloud_data[:, 0]/x_res)
        vx[:, 1] = np.floor(xyz_point_cloud_data[:, 1]/y_res)
        vx[:, 2] = np.floor(xyz_point_cloud_data[:, 2]/z_res)

        vx = vx[vx[:, 3].argsort()]
        voxelized, unique_indices = np.unique(vx, axis=0, return_index=True)
        voxelized = voxelized[np.argsort(unique_indices)]

        total_volume = voxelized.shape[0]*x_res*y_res*z_res

    return total_volume

# Helper function to calculate the total number of occupied voxels in a particular scene
# input_tuple: (pc,voxel_rsize, voxel_asize, voxel_esize, data,i)


def occupancy_count(input: tuple) -> np.float32:
    """Computes the ratio of occupied voxels to the total amount of
    voxels.

    Args:
        input (tuple): Tuple of form (pc, r_size, azimuth_size, elevation_size, sensorcon).
         - pc: The point cloud to voxelize.
         - r_size: Voxel size of the spherical coordinate; radius of sphere.
         - azimuth_size: Azimuth angle of each voxel from the center
         - elevation_size: Elevation angle of each voxel from the center
         - sensorcon: Our sensor configuration.
         - i: The frame in question being voxelized.
         - total_voxels: Total possible number of voxels (in cartesian or spherical coordinates)
         bounded by the FOV.

    Returns:
        out_ratio (np.float32): The ratio of occupied voxels to the the total number.
    """

    # Working with the Vista output point cloud

    pc = input[0]
    r_size = input[1]
    azimuth_size = input[2]
    elevation_size = input[3]
    sensorcon = input[4]
    i = input[5]
    total_voxels = input[6]

    # Filter out the x y z components
    xyz_point_cloud_data = pc[:, 0:3]

    if not USE_CARTESIAN:

        # Converting the point cloud into spherical coordinates
        azimuth, elevation, r = cart2sph(
            xyz_point_cloud_data[:, 0], xyz_point_cloud_data[:, 1], xyz_point_cloud_data[:, 2])

        # Converting from radians to Degrees
        # For both azimuth and elevation, both the min are now zero, and the max
        # has been changed from 180 to 360 for azimuth, and from 90 to 180 for
        # elevation. We basically got rid of negative angle and shifted the
        # notation by the respectable amount.
        azimuth = np.rad2deg(azimuth) + 180
        azimuth = np.mod(azimuth, 360)
        elevation = np.rad2deg(elevation) + 90
        elevation = np.mod(elevation, 180)
        spherical_point_cloud_data = np.transpose(
            np.array([r, elevation, azimuth]))

        # Origin coordinate setting. Where the sensor is. Move to outside the
        # function in the future.
        voxel_struct = {}
        voxel_struct["r_size"] = r_size
        voxel_struct["a_size"] = azimuth_size
        voxel_struct["e_size"] = elevation_size

        # Removing values that are over the constraints
        # Range
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 0] < sensorcon["r_high"]]
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 0] >= sensorcon["r_low"]]

        # Elevation range
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 1] < sensorcon["e_high"]]
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 1] >= sensorcon["e_low"]]

        # Azimuth range
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 2] < sensorcon["a_high"]]
        spherical_point_cloud_data = spherical_point_cloud_data[
            spherical_point_cloud_data[:, 2] >= sensorcon["a_low"]]

        # Spherical voxelization
        # Think of each voxel as a dV element in spherical coordinates, except that its
        # not of infinitesmal size. We are simply making an evenly spaced grid (in spherical coordinates)
        # of the solid that is bounded the sensor FOV.
        #
        # Here we take the real coordinates of each point, and convert to voxel indices
        # We take the floor of the voxel indices that are 'close enough' to each other;
        # i.e., duplicate indices correspond to multiple points within a voxel
        spherical_point_cloud_data[:, 2] = np.floor(
            spherical_point_cloud_data[:, 2]/voxel_struct["a_size"])
        spherical_point_cloud_data[:, 1] = np.floor(
            spherical_point_cloud_data[:, 1]/voxel_struct["e_size"])
        spherical_point_cloud_data[:, 0] = np.floor(
            spherical_point_cloud_data[:, 0]/voxel_struct["r_size"])

        # Only keep unique voxels
        # Removing duplicates creates the voxel data and also sorts it
        # We are also handling occlusions by sorting the original coordinates by
        # the distance from the sensor. Then we run unique on the azimuth and
        # elevation angle. The index output from that will be used to basically get
        # rid of all the voxels behind a certain angle coordinate.
        spherical_point_cloud_data = spherical_point_cloud_data[spherical_point_cloud_data[:, 0].argsort(
        )]
        vx, unique_indices = np.unique(
            spherical_point_cloud_data, axis=0, return_index=True)
        vx = vx[np.argsort(unique_indices)]

        # Pass total voxels from outside of the loop
        out_ratio = vx.shape[0] / total_voxels

    else:
        vx = np.zeros((xyz_point_cloud_data.shape[0], 4))

        # Fourth column is range which we will sort later on.
        vx[:, 3] = np.linalg.norm(xyz_point_cloud_data, axis=1)

        vx[:, 0] = np.floor(xyz_point_cloud_data[:, 0]/x_res)
        vx[:, 1] = np.floor(xyz_point_cloud_data[:, 1]/y_res)
        vx[:, 2] = np.floor(xyz_point_cloud_data[:, 2]/z_res)

        vx = vx[vx[:, 3].argsort()]
        voxelized, unique_indices = np.unique(vx, axis=0, return_index=True)
        voxelized = voxelized[np.argsort(unique_indices)]

        # Pass total voxels from outside the loop
        out_ratio = voxelized.shape[0] / total_voxels

    return out_ratio

### Driver functions below ###


def data_rate_vista_automated(
    sensorcon_path: str,
    vistaoutput_path: str,
    prepad_output: bool = True,
    enable_graphical: bool = True,
    enable_regression: bool = True,
    regression_power: int = 9,
    enable_resolution: bool = False,
    resolution: int = 1
) -> None:

    f = open(sensorcon_path)
    data = json.load(f)

    # Making sensor parameters so that we can find the accurate volume ratio.
    data["e_high"] += 90
    data["e_low"] += 90
    data["a_high"] += 180
    data["a_low"] += 180

    sensor_range = data["r_high"]  # Meters.
    padding_size = sensor_range  # Meters.

    # Set spherical voxel size from sensor precision:
    voxel_asize = data["horizAngRes"]
    voxel_esize = data["verticAngRes"]
    azimuth_fov = data["a_high"] - data["a_low"]
    elevation_fov = data["e_high"] - data["e_low"]

    # Refresh rate of the sensor. Needed for calculations.
    refresh_rate = 20  # In hertz

    # SNR is the signal to noise ratio. I don't know where to find these values.
    snrMax = 12  # I don't know the correct ballpark for this value.

    # Bit per measurements
    bitspermeasurements = 12  # Unit of bit.

    # just the r precision, which should be chosen by the user.
    voxel_rsize = 0.03    # in meters. Edit if you need to.

    point_density = 1

    file_regex = "output_*.txt"

    # Calculate the total volume bounded by the sensor's FOV. This is a simple volume integral in spherical coordinates.
    max_volume = (1/3)*(math.pow(data["r_high"], 3) - math.pow(data["r_low"], 3))\
        * (np.cos(np.deg2rad(data["e_low"]))-np.cos(np.deg2rad(data["e_high"])))\
        * (np.deg2rad(data["a_high"])-np.deg2rad(data["a_low"]))

    numScenes = len(vistaoutput_path)

    path = []
    total_scenes = []
    observers = []
    outmatrix_volume = []
    outmatrix_count = []

    for i in range(numScenes):
        path.append(os.path.join(vistaoutput_path[i], file_regex))

        total_scenes.append(len(glob.glob(path[-1])))

        # total_scenes is how many scenes there are in a folder
        # observers is how many scenes we intend to analyse
        if enable_resolution:
            # integer divison to round down so that you wont get extra observers that might crash the program
            observers.append(total_scenes[-1] // resolution)
        else:
            observers.append(total_scenes[-1])

        # THIS IS IF YOU HAVE RENAMED THE NUMBERING OF THE VISTA OUTPUTS
        # SET TO ZERO IF OUTPUT NAMES OF FILES WERE NOT RENAMED IN ANYWAY
        global_offset = 0

        outmatrix_volume.append(np.zeros([observers[-1]+global_offset, 2]))
        outmatrix_count.append(np.zeros([observers[-1]+global_offset, 2]))

    numCores = mp.cpu_count() - 1

    for itr in range(numScenes):
        print('\nWorking on scene: ' + str(vistaoutput_path[itr]))
        # Read each of the scenes into memory in parallel, but can be configured to read once every few scenes
        # find smallest frame number in folder, program assumes smallest frame will be used in the graph analysis
        # need to find this so that program knows at which index to insert data into outmatrix
        smallest = math.inf
        for filename in os.listdir(vistaoutput_path[itr]):
            numFrame = get_first_number(filename)
            if smallest > numFrame:
                smallest = numFrame
        upperbound = smallest + observers[itr] * resolution

        # Calculate deltas using the volume method if selected.
        if USE_VOLUMETRIC:
            results_list_path = os.path.join(os.getcwd(), 'results_vol.pkl')
            results_vol = None
            if not os.path.exists(results_list_path):
                with mp.Pool(numCores) as p:
                    inputData = [(voxel_rsize, voxel_asize, voxel_esize, data, i, vistaoutput_path[itr],
                                  point_density, max_volume, global_offset) for i in range(smallest, upperbound, resolution)]
                    results = []
                    with tqdm(total=len(inputData), desc="Processing Volume") as pbar:
                        for result in p.imap(multiprocessed_vol_funct, inputData):
                            # result[0] is i in line 119. This subtraction is done so that result[1] can be inserted into
                            # outmatrix_volume at the proper index, which starts from 0, rather than at the variable "smallest"
                            outmatrix_volume[itr][(
                                result[0] - smallest)//resolution] = result[1]
                            results.append(result)
                            pbar.update()

                with open(results_list_path, "wb") as f:
                    pickle.dump(results, f)
                    results_vol = results
            else:
                print("Loading saved list...")
                with open(results_list_path, "rb") as f:
                    results = pickle.load(f)
                    for result in results:
                        outmatrix_volume[itr][(
                            result[0] - smallest)//resolution] = result[1]

                    # print("in else", type(scenes))

        # Calculate deltas using the simple method.
        # Calculate the total possible number of Cartesian voxels within our FOV given in spherical coordinates.
        if USE_CARTESIAN:
            voxel_volume_cart = x_res * y_res * z_res
            # We are only interested in the number of voxels that fit inside our volume, not where.
            total_voxels = np.floor(max_volume / voxel_volume_cart)
        else:
            # Simply obtain the total number of spherical voxels.
            # Finding the total amount of voxels in this range with config
            azimuth_capacity = np.floor(
                (data["a_high"]-data["a_low"])/data["a_size"])
            elevation_capacity = np.floor(
                (data["e_high"]-data["e_low"])/data["e_size"])
            radius_capacity = np.floor(
                (data["r_high"]-data["r_low"])/data["r_size"])
            total_voxels = azimuth_capacity * elevation_capacity * radius_capacity

        results_list_path = os.path.join(os.getcwd(), 'results_cart.pkl')
        results_cart = None
        if not os.path.exists(results_list_path):
            with mp.Pool(numCores) as p:
                inputData = [(voxel_rsize, voxel_asize, voxel_esize, data, i, vistaoutput_path[itr],
                              point_density, total_voxels, global_offset) for i in range(smallest, upperbound, resolution)]
                results = []
                with tqdm(total=len(inputData), desc="Processing Count") as pbar:
                    for result in p.imap(multiprocessed_count_funct, inputData):
                        # result[0] is i in line 133. This subtraction is done so that result[1] can be inserted into
                        # outmatrix_count at the proper index, which starts from 0, rather than at the variable "smallest"
                        outmatrix_count[itr][(
                            result[0] - smallest)//resolution] = result[1]
                        results.append(result)
                        pbar.update()
            with open(results_list_path, "wb") as f:
                pickle.dump(results, f)
                results_cart = results
        else:
            print("Loading saved list...")
            with open(results_list_path, "rb") as f:
                results = pickle.load(f)
                for result in results:
                    outmatrix_count[itr][(
                        result[0] - smallest)//resolution] = result[1]

    print('\nDone!')

    # Calculate delta/deltamax for volumetric method if we are using it
    if USE_VOLUMETRIC:
        outmatrix_volume2 = copy.deepcopy(outmatrix_volume)
        outmatrix_count2 = copy.deepcopy(outmatrix_count)

        for i in range(numScenes):
            outmatrix_volume2[i][:, 1] = outmatrix_volume2[i][:,
                                                              1] / np.max(outmatrix_volume2[i][:, 1])  # BANDAID FIX
            outmatrix_count2[i][:, 1] = outmatrix_count2[i][:,
                                                            1] / np.max(outmatrix_count2[i][:, 1])

    # calculating rolling average
    def rolling_average(input, col):
        window = 10
        average_y = []

        # for j in range(numScenes):
        for ind in range(len(input[:, col]) - window + 1):
            average_y.append(np.mean(input[ind:ind+window, col]))
        # print(len(average_y))

        for ind in range(window - 1):
            average_y.insert(0, np.nan)

        return average_y

    if USE_VOLUMETRIC:
        outmatrix_volume_ave = []
        outmatrix_volume2_ave = []
        outmatrix_count2_ave = []

    outmatrix_count_ave = []

    # Get rolling averages for all deltas
    for itr in range(numScenes):
        if USE_VOLUMETRIC:
            outmatrix_volume_ave.append(
                rolling_average(outmatrix_volume[itr], 1))
            outmatrix_volume2_ave.append(
                rolling_average(outmatrix_volume2[itr], 1))
            outmatrix_count2_ave.append(
                rolling_average(outmatrix_count2[itr], 1))
        else:
            pass
        outmatrix_count_ave.append(rolling_average(outmatrix_count[itr], 1))

    datarate_buffer = (32*sensor_range*azimuth_fov*elevation_fov*refresh_rate*bitspermeasurements)\
        / (3*voxel_asize*voxel_esize*voxel_rsize*snrMax)

    an_data_rate = []
    an_data_rate2 = []

    for i in range(numScenes):
        # Calculate data rate from volumetric method, if selected.
        if USE_VOLUMETRIC:
            log_inverse_delta = np.log((1/(2*outmatrix_volume[i][:, 1])))
            delta_log_inverse_delta = outmatrix_volume[i][:,
                                                          1] * log_inverse_delta
            an_data_rate.append(np.transpose(
                np.array([delta_log_inverse_delta * datarate_buffer])))

        # Calculate data rate from simple method.
        log_inverse_delta2 = np.log((1/(2*outmatrix_count[i][:, 1])))
        delta_log_inverse_delta2 = outmatrix_count[i][:,
                                                      1] * log_inverse_delta2
        an_data_rate2.append(np.transpose(
            np.array([delta_log_inverse_delta2 * datarate_buffer])))

    # Print the raw data rate values
    for scene_number in range(numScenes):
        output_folder = os.path.join(
            "data_rates", os.path.basename(vistaoutput_path[scene_number]))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        np.savetxt(os.path.join(output_folder, "an_data_rate_count.csv"),
                   np.concatenate(
                       (outmatrix_count[scene_number][:, 0][:, np.newaxis], an_data_rate2[scene_number]), axis=1),
                   delimiter=',')
        np.savetxt(os.path.join(output_folder, "data_ratio_count.csv"),
                   outmatrix_count[scene_number],
                   delimiter=',')

        if USE_VOLUMETRIC:
            # We are ignoring the volume method but I included it just because.
            np.savetxt(os.path.join(output_folder, "an_data_rate_volume.csv"),
                       np.concatenate(
                           (outmatrix_volume2[scene_number][:, 0][:, np.newaxis], an_data_rate[scene_number]), axis=1),
                       delimiter=',')
            np.savetxt(os.path.join(output_folder, "data_ratio_volume.csv"),
                       outmatrix_volume[scene_number],
                       delimiter=',')

            np.savetxt(os.path.join(output_folder, "delta_deltamax_count.csv"),
                       outmatrix_count2[scene_number],
                       delimiter=',')
            np.savetxt(os.path.join(output_folder, "delta_deltamax_volume.csv"),
                       outmatrix_volume2[scene_number],
                       delimiter=',')
        else:
            pass

    an_data_rate_ave = []
    an_data_rate2_ave = []

    for itr in range(numScenes):
        # Calculate the rolling average of the data rate if selected.
        if USE_VOLUMETRIC:
            # BANDAID FIX (we do not care about the volume method for now)
            an_data_rate_ave.append(rolling_average(an_data_rate[itr], 0))
        else:
            pass
        an_data_rate2_ave.append(rolling_average(an_data_rate2[itr], 0))

    # green is for simple, red is for volumetric
    # complementary_colours = [['-r','-c'],['-g','-m'],['-b','-y']]
    def showDataRateGraph(xBarData, yBarData, yBarAverageData, windowTitle, graphTitle, xlabel, ylabel, isSimple):
        if isSimple:
            colourScheme = [['g', 'm'], ['b', 'y']]
        else:
            colourScheme = [['r', 'c'], ['b', 'y']]

        fig4, ax4 = plt.subplots()
        fig4.canvas.manager.set_window_title(f'{windowTitle}')
        fig4.suptitle(f"{graphTitle}", fontsize=12)
        ax4.set_ylabel(f"{ylabel}", color='black')
        ax4.tick_params(axis='y', colors='black')
        for i in range(numScenes):
            if i == 0:
                # ORIGINAL PLOT
                ax4.plot(xBarData[i][:, 0], yBarAverageData[i],
                         f'{colourScheme[np.mod(i,2)][1]}')
            else:
                print('else called')
                ax4_new = ax4.twinx()
                ax4_new.plot(xBarData[i][:, 0], yBarAverageData[i],
                             f'{colourScheme[np.mod(i,2)][1]}')

                ax4_new.tick_params(
                    axis='y', colors=f'{colourScheme[np.mod(i,2)][0]}')

                offset = (i - 1) * 0.7
                ax4_new.spines['right'].set_position(('outward', offset * 100))

        ax4.set_xlabel(f"{xlabel}, huh")
        fig4.legend()
        fig4.tight_layout()
        cursor(hover=True)

        return fig4, ax4

    def displayDataRateGraph(xBarData, yBarData, yBarAverageData, windowTitle, graphTitle, xlabel, ylabel, isShowOriginal, isShowAverage, isShowRegression):
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(f'{windowTitle}')
        fig.suptitle(f"{graphTitle}", fontsize=12)
        ax.set_ylabel(f"{ylabel} {get_folder(vistaoutput_path[0])}", color='r')
        ax.tick_params(axis='y', colors='r')
        for i in range(numScenes):
            if i == 0:
                # ORIGINAL PLOT
                ax.plot(xBarData[i][:, 0], yBarData[i][:, 0],
                        f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                # ROLLING AVERAGE
                ax.plot(xBarData[i][:, 0], yBarAverageData[i],
                        f'g', label=f'Rolling Average')
                # BEST FIT LINE
                poly, residual, _, _, _ = np.polyfit(
                    xBarData[i][:, 0], yBarData[i][:, 0], deg=regression_power, full=True)
                ax.plot(xBarData[i][:, 0], np.polyval(poly, xBarData[i][:, 0]),
                        f'b',
                        label=f'Fitted: {get_folder(vistaoutput_path[i])}')
            else:
                ax_new = ax.twinx()
                # ORIGINAL PLOT
                ax_new.plot(xBarData[i][:, 0], yBarData[i][:, 0],
                            f'r', label=f'Original: {get_folder(vistaoutput_path[i])}')
                # ROLLING AVERAGE
                ax_new.plot(xBarData[i][:, 0], yBarAverageData[i],
                            f'g', label=f'Rolling Average')
                # BEST FIT LINE
                poly, residual, _, _, _ = np.polyfit(
                    xBarData[i][:, 0], yBarData[i][:, 0], deg=regression_power, full=True)
                ax_new.plot(xBarData[i][:, 0], np.polyval(poly, xBarData[i][:, 0]),
                            f'b',
                            label=f'Fitted: {get_folder(vistaoutput_path[i])}')
                # Setting new Y-axis
                ax_new.set_ylabel(
                    f"Atomic norm Data rate {get_folder(vistaoutput_path[i])}", color='black')
                ax_new.tick_params(
                    axis='y', colors=complementary_colours[np.mod(i, 3)][0])

                offset = (i - 1) * 1
                ax_new.spines['right'].set_position(('outward', offset * 100))

        ax.set_xlabel(f"{xlabel}")
        # plt.ylabel("Atomic norm Data rate")
        fig.legend()
        fig.tight_layout()

        return fig, ax

    def saveGraphImages(xBarData, yBarData, yBarAverageData, windowTitle, graphTitle, xlabel, ylabel, isSimple):
        fig, ax = plt.subplots()
        t = np.linspace(0, 3, 40)
        g = -9.81
        v0 = 12
        z = g * t**2 / 2 + v0 * t

        v02 = 5
        z2 = g * t**2 / 2 + v02 * t

        scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
        line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
        ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
        ax.legend()

        plt_images_dir = os.path.join(os.getcwd(), "plt_images")
        if not os.path.exists(plt_images_dir):
            os.makedirs(plt_images_dir)

        def update(frame):
            # for each frame, update the data stored on each artist.
            x = t[:frame]
            y = z[:frame]
            # update the scatter plot:
            data = np.stack([x, y]).T
            scat.set_offsets(data)
            # update the line plot:
            line2.set_xdata(t[:frame])
            line2.set_ydata(z2[:frame])
            plt.savefig(os.path.join(plt_images_dir, f'frame_{len(x)}.png'))
            return (scat, line2)

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=40, interval=30, repeat=False)
        plt.show()

    # Datarate graphs
    if not enable_graphical:
        if enable_regression:
            # Get data rate plots for simple method
            fig53, ax53 = showDataRateGraph(outmatrix_count, an_data_rate2,
                                            an_data_rate2_ave, 'Simple method datarate', 'Data rate for occupancy count', 'distance (m)',
                                            'Atomic norm Data rate', True)

            # Get data rate plots for volume method
            if USE_VOLUMETRIC:
                fig43, ax43 = showDataRateGraph(outmatrix_count, an_data_rate,
                                                an_data_rate_ave, 'Volume method datarate', 'Data rate for volumetric method', 'distance (m)',
                                                'Atomic norm Data rate', True)
        else:
            # TO UPDATE
            '''
            fig4 = plt.figure("Volume method datarate")
            fig4.suptitle("Data rate of volumetric voxelization method", fontsize=12)
            for i in range(numScenes):
                plt.plot(outmatrix_volume[i][:, 0], an_data_rate[i][:, 0], f'{complementary_colours[np.mod(i,3)][0]}')
            plt.xlabel("distance (m)")
            plt.ylabel("Atomic norm Data rate")

            #plt.show(block=False)
            #plt.show()

            fig5 = plt.figure("Simple method datarate")
            fig5.suptitle("Data rate of simple voxelization method", fontsize=12)
            for i in range(numScenes):
                plt.plot(outmatrix_count[i][:, 0], an_data_rate2[i][:, 0], f'{complementary_colours[np.mod(i,3)][0]}')
            plt.xlabel("distance (m)")
            plt.ylabel("Atomic norm Data rate")        


            #plt.show(block=False)   
            #plt.show()    
            '''
    plt.show()
    print("done saving")


def main():
    args = file_tools.parse_cmdline_args()
    sensorcon_path = file_tools.obtain_sensor_path(args)
    path2scenes = [os.path.abspath(os.environ["VISTA_OUTPUT_PATH"])]

    data_rate_vista_automated(
        sensorcon_path=sensorcon_path,
        vistaoutput_path=path2scenes,
        prepad_output=True,
        enable_graphical=True,
        enable_regression=True,
        regression_power=10
    )
    return


if __name__ == "__main__":
    main()
