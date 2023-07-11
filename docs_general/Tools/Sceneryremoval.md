# Scenery removal documentation

[Back to README.md](../README.md)

## Overview

This is if you want to preprocess the input .las section such that only (or most of) the pavement remains; removing vegetation, scenery, etc. The program cuts out a fixed width (for now) along the trajectory from the input .las road section. Automatically determining the road boundaries is a bit more of a complex task, so for now we are cutting a fixed width for every road point.

### Methodology

From our trajectory with $N$ road points, we define $N$ 2D bounding boxes of a certain width and length that are then translated and rotated at each road point using the forward, leftward, and upward vectors from the [trajectory](Trajectory.md). By translating and rotating the $N$ bounding boxes to each road point, we are defining the region of the road that we will cut out.

After translating these bounding boxes, we will end up with something along the lines of this:

![Example of the bounding boxes translated/centered at the road points](images/example_bounding_region.png)

*The colored path of points represent the road points. In this picture, the bounding boxes were set to 1.9m left of the road point, and 5.7m right of the road point.*

> *We generate each bounding box with a different coordinate system (i.e., Cartesian), and then use the trajectory to center each bounding box for each road point.*

where we end up with a path of bounding points made from the bounding boxes centered at the road points. We can then perform a test *(see the code in the ``slice_road_from_bound()`` method of ``remove_scenery.py``)* to see if the las file's points are inside a single bounding box, for each bounding box, and identify the indices (which points) of the .las file are within a single bounding box.

Here's an example of a single bounding box, along the path of road points in blue. The point that our box is centered about is larger.

![Example of a single bounding region](images/example_single_bounding_region.png)
*The larger point represents the origin from the local coordinate system that we defined the bounding box.*

This process is done for the rest of the bounding boxes, leaving us with overlap in the indices calculated. We then take unique indices, and then we have our .las file cut to our width as seen in blue:

![Example of all of the points, trimmed](images/example_bounding_region_with_trimmed.png)

*Note: When working with large .las inputs (>35 million points), this process will require a lot of memory (64GB is recommended).*

#### Variable road widths

As of 27/6/2023, the current implementation only cuts the road with a fixed width from the road points, making it only really viable with rural road sections as they are more consistent in terms of lanes and whatnot.

The ``generate_bounds()`` method in ``remove_scenery.py`` allows for variable widths to be inputted into the code, however, the road boundary needs to be automatically detected for variable road widths to work.

Here is a code snippet of the method:

```python
# Generate bounding boxes for every road point, with a specified width and length
# at every road point
def generate_bounds(traj: file_tools.Trajectory) -> np.ndarray:
    # Generates bounding boxes for each road point
    # Note that widths, lengths, are both (N, 2) arrays, where 
    # the y-direction for the bounds left and right of a certain
    # road point can be changed, allowing for variable widths.

    num_points = traj.getNumPoints()
    # 5m left and right of the observer
    width = np.array([-5, 5]).reshape((1, 2))       # In the y-direction 
    widths = np.repeat(width, num_points, axis=0)
    # 1.5m behind and in front of the observer
    length = np.array([-1.5, 1.5]).reshape((1, 2))  # In the x-direction
    lengths = np.repeat(length, num_points, axis=0) # Repeats the lengths

    return widths, lengths
```

#### Removing vehicles/noise in the z-coordinate from the scenery

Sometimes, points that resemble vehicles can appear in the cut down .las file, causing noise. When trimming the road, note the current method mentioned above trims out points **regardless of z-coordinate**. In the code, it is mentioned that each bounding box is 'projected' into the xy-plane of of the .las file by ignoring the z-coordinate when testing if the points in the .las file are inside a bounding box. Thus, .las files with vehicles on the road (i.e., vehicles where parked, and scanned as the scanning vehicle passed by) may appear when the current implementation is ran.

The ``scenery3d`` branch aims to solve this issue by using 3D bounding boxes instead of 2D bounding boxes, as well as a different test. However, at its current implementation right now, I think that you are better off manually removing the vehicles from the road using CloudCompare. Manually cleaning the road sections is probably more effective, as you do not have to fiddle with the dimensions of the bounding box.

Example of the points that make up the 3D bounding region:

![Example of 3D bounding points](images/example_3d_bounding_region.png)

*The bottom bounds of the far side of the road points are faint, but visible. In this case for the z-coordinate, we take 0.25m above the road point and 2m below the road point to remove noise points such as vehicles.*

Here is the cut road section, with the 3D method in red.

![Example of the cut road, using the 3D method](images/example_3d_bounding_region_with_trimmed.png)

Note that the 3D method (in red) appears a lot more jagged as it is right now. As I mentioned earlier, you are probably better off cutting the road using the 2D method and manually removing vehicles using CloudCompare.

![Example of the cut road, with the jagged edges for some reason](images/example_3d_trimmed_jagged.png)

The bounding points and road points are shown again for reference.
