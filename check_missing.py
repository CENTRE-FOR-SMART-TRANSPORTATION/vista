import os, sys
from pathlib import Path
import argparse
import glob


'''
Sometimes, with the current shell script automation, some processes
do not output the full range of frames that they were assigned to,
leading to some missing output scenes. This program is here just in case
if you want to identify the missing outputs that happened for some reason.
'''

# Global variables for file I/O
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
ROOT2 = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def identify_missing_filenames(args: argparse.Namespace) -> None:
  """Identifies the missing outputs within a specific range. 
  If a range is not specified, then it will default to the first and last frame. 

  Args:
      args (argparse.Namespace): Arguments passed from the command line.
  """

  ## Obtain input folder path to check
  if args.output == None:
    import tkinter as tk
    from tkinter import Tk
    import tkinter.filedialog
    
    # Manually open trajectory folder
    Tk().withdraw()
    scenes_folderpath = tk.filedialog.askdirectory(
      initialdir=ROOT2, title="Please select the Vista output folder"
    )
    print(
      f"\nYou have chosen to open the folder to the scenes:\n{scenes_folderpath}"
    )

  else:
    # Use trajectory folder from defined command line argument
    scenes_folderpath = args.output


  ## Obtain consecutive range of frames found in the output directory
  abs_paths = glob.glob(os.path.join(scenes_folderpath, "*.txt"))
  filenames = [os.path.basename(p) for p in abs_paths]
  
  # startframe and endframe are automatically set if not provided
  if (args.startframe == None and args.endframe == None):
    # Automatically set startframe and endframe to the max and min filename numbers
    startframe = int(min(filenames, key=lambda x: int((x.split('_'))[1])).split('_')[1])
    endframe = int(max(filenames, key=lambda x: int((x.split('_'))[1])).split('_')[1])
  else:
    startframe = args.startframe
    endframe = args.endframe
  
  
  ## Just get the resolution of the outputs
  res = [float(os.path.splitext((fname.split("_")[-1]))[0]) for fname in filenames]
  assert len(set(res)) == 1, "Outputs have unequal resolution!"
  res = res[0]


  ## Now we can obtain our range
  frames = []
  framerange = range(startframe, endframe+1)
  
  for p in abs_paths:
    frame = int(os.path.basename(p).split('_')[1])
    
    if frame in framerange:
      frames.append(frame)
    
  frames = sorted(frames)


  ## Then find the missing integers in our frame range
  missing_frames = sorted(set(range(frames[0], frames[-1])) - set(frames))
  
  if len(missing_frames == 0):
    print("There are no missing outputs.")
    return
  
  print(f"Outputs were missing from frame range {startframe} to {endframe}:")
  for missing_frame in missing_frames:
    print(f"output_{missing_frame}_{res}.txt")
    
  return


def main() -> None:
  args = parse_args()
  identify_missing_filenames(args)
  
  return

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--startframe", type=int, default=None, help="Specified start frame of the Vista scenes")
  parser.add_argument("--endframe", type=int, default=None, help="Specified start frame of the Vista scenes")
  parser.add_argument("--output", type=str, default=None, help="Path to the Vista output folder")

  return parser.parse_args()

if __name__ == "__main__":
  main()