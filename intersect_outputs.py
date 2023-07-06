import numpy as np
import pandas as pd
import open3d as o3d
import dask.dataframe as dd
import glob
import time
from tkinter.simpledialog import askinteger

import file_tools

import sys, os
from pathlib import Path

# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


# Performs intersection between two point clouds
def intersect(pcd0: np.ndarray, pcd1: np.ndarray, tol: np.float32) -> np.ndarray:
  """Finds the intersection of points given two point clouds ``pcd0`` and ``pcd1``
  within a tolerance of ``tol``mm.
  
  Args:
      pcd0 (np.ndarray): First point cloud.
      pcd1 (np.ndarray): Second point cloud.
      tol (np.float32): Tolerance allowed for intersecting points of the point cloud.

  Returns:
      intersected (np.ndarray): Intersected point cloud. 
  """
  
  # We define pcd0 to the the smaller point cloud and pcd1 to be the larger one
  # This just enforces the notation that I have specified
  if pcd0.shape[0] > pcd1.shape[0]:
    temp_pcd = pcd1 # buffer
    pcd0 = pcd1
    pcd1 = temp_pcd
    del temp_pcd
    
  o3d_pcd0 = o3d.geometry.PointCloud()
  o3d_pcd0.points = o3d.utility.Vector3dVector(pcd0[:, 0:3])
  
  o3d_pcd1 = o3d.geometry.PointCloud()
  o3d_pcd1.points = o3d.utility.Vector3dVector(pcd1[:, 0:3])
  
  # For each point of the larger point cloud, we will compute the shortest distance to the nearest neighbour (1)
  # of the smaller point cloud, using a KNN search provided by o3d.PointCloud.compute_point_cloud_distance().
  distances2smaller_pcd = np.asarray(o3d_pcd1.compute_point_cloud_distance(target=o3d_pcd0))
  
  # Now take the points of the source point that are close enough for intersection
  intersected = pcd1[distances2smaller_pcd < tol]

  return intersected

# Helpter function to write a single intersected point cloud to csv
def write_output(path2output: str, data: np.ndarray) -> None:
  df = pd.DataFrame(data)
  df.columns = ['x', 'y', 'z', 'yaw', 'pitch', 'depth']
  df.to_csv(path2output, index=False)
  return

# parallel for loop to run intersections goes here
def run_intersections(dfs: list, offset: int, res: np.float32, src_name: str, framerange: list, tol: np.float32) -> None:
  
  import joblib
  from joblib import Parallel, delayed
  from tqdm import tqdm
  
  total_scenes = framerange[1] - framerange[0]
  cores = (joblib.cpu_count() - 1)

  print("")
  # Define the arguments that will be called upon in parallel
  arguments = [
    (np.asarray(dfs[0].get_partition(frame).to_dask_array()), # pcd0, in ndarray format
     np.asarray(dfs[1].get_partition(frame).to_dask_array()), # pcd1, in ndarray format
     tol,
     )
    
    for frame in tqdm(range(framerange[0], framerange[1]), desc="Reading point clouds into memory")
    ]
  print("Inputs successfully read into memory.")
  
  print("")
  tStart = time.perf_counter()
  intersected = Parallel(n_jobs=cores, backend='loky')( # Switched to loky backend to maybe suppress errors?
      delayed(intersect)(arg_pcd0, arg_pcd1, arg_tol)
        for arg_pcd0, arg_pcd1, arg_tol in tqdm(arguments, 
                                                total=total_scenes,
                                                desc="Intersecting scenes")
      )
  
  tStop = time.perf_counter()
  print(f"Intersection complete in {tStop-tStart:.2f}s.")
  
  # Write our outputs
  output_dir = os.path.join(ROOT2, "compared")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  output_subdir = os.path.join(output_dir, f"{src_name}_intersected_resolution={res:.2f}")
  if not os.path.exists(output_subdir):
    os.makedirs(output_subdir)

  print("")
  # Multiprocessed pandas to csv because I couldn't get dask to write multiple .csv files.
  Parallel(n_jobs=cores, backend='loky')(
    delayed(write_output)(os.path.join(output_subdir, f"output_{i+framerange[0]+offset}_{res:.2f}.txt"), intersected[i])
    for i in tqdm(range(total_scenes),
                  total=total_scenes,
                  desc="Writing output")
  )
  print(f"Output written to {output_subdir}.")

  return

# just gets the frame offset for padding, as well as the sensor res
def obtain_output_details(paths2outputs_ext: list) -> int or np.float32:
  
  # Obtain filenames, as well as total number files for each respective directory
  # It is assumed that both input folders have the same number of files
  filenames = []
  total_files = []
  for i in range(2):
    filenames_per_scene = [os.path.basename(abs_path) for abs_path in glob.glob(paths2outputs_ext[i])]
    # sort the filenames per scene because glob is kind of wacky
    filenames_per_scene = sorted(filenames_per_scene, key=lambda f: int(f.split('_')[1])) 
    filenames.append(filenames_per_scene)
    total_files.append(len(filenames_per_scene))
    
    
  # Obtain sensor range, in m for padding
  offsets = [int(min(filenames[i], key=lambda x: int((x.split('_'))[1])).split('_')[1]) for i in range(2)] # this just gets the min value for the frame
  # Make sure that both offsets are equal
  assert len(set(offsets)) == 1, "Offsets are not equal, are you sure that the outputs selected are generated properly?"
  assert len(set(total_files)) == 1, "Total number of files are not equal, are you sure that you are missing some outputs or are you trying to compare two different roads?"
  
  offset = offsets[0]
  total = total_files[0]
  
  res = np.float32(
        float(os.path.splitext(
            (filenames[0][0].split("_")[-1])   
        )[0])
      )
  
  return offset, res, total

# Read Vista outputs into Dask DataFrames for parallelism
def read_inputs(paths2outputs_ext: list) -> list:

  # DataFrames for each output folder
  dfs = []
  for i in range(2):

    # Absolute paths of filenames will be read into the DataFrame's partitions
    abs_paths = glob.glob(paths2outputs_ext[i])
    # sort the filenames per scene because glob is kind of wacky
    abs_paths = sorted(abs_paths, key=lambda f: int(os.path.basename(f).split('_')[1])) 
    
    # Dask has parallel I/O which is why I chose it
    dfs.append(dd.read_csv(abs_paths, skiprows=0))

  return dfs

# Obtain range to intersect from the user
def obtain_intersect_range(total_scenes: int, offset: int) -> list:
  # total_scenes corresponds to the number of files

  # frame0 corresponds to the filename, not the number of files
  # subtract offset from frame0 to obtain the file 'index'
  frame0 = askinteger(
    title="Enter the start frame",
    prompt=f"Enter the frame where you want to start intersecting FROM; inclusive.\n(Outputs numbered {0+offset}-{total_scenes+offset-1})"
    )
  
  assert ((frame0-offset) > 0 and (frame0-offset) < total_scenes), "Start frame must be in a valid range!"
  
  # frame1 corresponds to the filename, not the number of files
  # subtract offset from frame1 to obtain the file 'index'
  frame1 = askinteger(
    title="Enter the stop frame",
    prompt=f"Enter the frame where you want to intersect UP TO; not inclusive.\n(Outputs numbered {frame0}-{total_scenes+offset-1})"
    )
  
  assert ((frame1-offset) > 0 and (frame1-offset) < total_scenes), "Stop frame must be in a valid range!"
  
  # Just makes sure that the frame range in ascending order 
  # TODO Customize prompts if user chooses to input a higher value first and then a lower value
  
  framerange = sorted((frame0-offset, frame1-offset))
  print(f"\nOutputs numbered {framerange[0]+offset} to {framerange[1]+offset-1} will be intersected.")
  
  return framerange

def main():
  args = file_tools.parse_cmdline_args()
  args.numScenes = 2 # There will only be two folders that we will intersect 
  # TODO Modify file_tools to take scene paths through command line, right now it only takes one filepath
  
  
  # Obtain our paths to outputs
  paths2outputs = file_tools.obtain_multiple_scene_path(args)
  paths2outputs = [os.path.normpath(p) for p in paths2outputs] # clean up the path

  paths2outputs_ext = [os.path.join(paths2outputs[i], '*.txt') for i in range(2)]
  paths2outputs_ext = [os.path.normpath(p) for p in paths2outputs_ext] # clean up
  
  # Just get the offset (for padding) and resolution
  offset, res, total_scenes = obtain_output_details(paths2outputs_ext)
  
  # Get range for when to intersect our outputs
  framerange = obtain_intersect_range(total_scenes, offset)
  
  # Obtain dask dataframes from both outputs because it uses parallelism
  dfs = read_inputs(paths2outputs_ext)
  
  # Get source file name
  src_name = os.path.commonprefix([os.path.basename(p) for p in paths2outputs])

  # Finally runs the intersections and writes the output
  # We allow a tolerance of one mm (or units that the data was given in) for an intersection
  run_intersections(dfs, offset, res, src_name, framerange, tol=1)
  
  return

if __name__ == "__main__":
  main()