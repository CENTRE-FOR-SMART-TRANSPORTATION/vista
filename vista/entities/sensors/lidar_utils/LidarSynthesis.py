import os
import h5py
import imp
import importlib.resources as pkg_resources
import numpy as np
from scipy import interpolate
from typing import Tuple, Optional, Union
import torch
import warnings

import vista
from vista import resources
from vista.utils import transform, logging
from .Pointcloud import Pointcloud, Point
from .s2d_model import LidarModel

tensor_or_ndarray = Union[torch.Tensor, np.ndarray]


class LidarSynthesis:
    """A Lidar synthesizer that simulates point cloud from novel viewpoint around
    a pre-collected Lidar sweep. At a high level, it involves (1) performing rigid
    transformation on point cloud based on given viewpoint change (2) projecting 3D
    point cloud to 2D image space with angle coordinates (3) densifying the sparse
    2D image (4) culling occluded region (5) masking out some points/pixels to simulate
    the sparse pattern of LiDAR sweep (6) reprojecting back to 3D point cloud or rays.

    Args:
        input_yaw_fov (float): Input LiDAR field of view in yaw axis; can be read from ``params.xml`` file.
        input_pitch_fov (float): Input LiDAR field of view in pitch axis; can be read from ``params.xml`` file.
        yaw_fov (float): Output LiDAR field of view in yaw axis; default is ``input_yaw_fov``.
        pitch_fov (float): Output LiDAR field of view in pitch axis; default is ``input_pitch_fov``.
        yaw_res (float): Resolution in yaw axis; default is ``0.1``.
        pitch_res (float): Resolution in pitch axis; default is ``0.1``.
        culling_r (int): The radius (from the origin) for culling occluded points.
        load_model (bool): Whether to load Lidar densifier model; default to ``True``.

    """

    def __init__(
        self,
        frame: int,
        input_yaw_fov: Tuple[float, float],
        input_pitch_fov: Tuple[float, float],
        yaw_fov: Optional[Tuple[float, float]] = None,
        pitch_fov: Optional[Tuple[float, float]] = None,
        yaw_res: float = 0.11,
        pitch_res: float = 0.11,
        culling_r: int = 2,
        load_model: bool = True,
        downsample: bool = False,
        **kwargs,
    ):
        # Filename passed from the shell script
        self.roadsection_filename = os.path.splitext(kwargs["roadsection_filename"])[0]
        self._frame = frame
        self._downsample = downsample

        ### Basic properties required for setting up the synthesizer including
        # the dimensionality and resolution of the image representation space
        self._res = np.array([yaw_res, pitch_res], dtype=np.float32)
        #self._fov = np.array([input_yaw_fov, input_pitch_fov], dtype=np.float32)
        self._fov = np.array([yaw_fov, pitch_fov], dtype=np.float32) # Called from the shell script itself
        self._fov_rad = self._fov * np.pi / 180.0

        self._dims = (self._fov[:, 1] - self._fov[:, 0]) / self._res
        self._dims = np.ceil(self._dims).astype(int)[:, np.newaxis]

        yaw_fov = input_yaw_fov if yaw_fov is None else yaw_fov
        pitch_fov = input_pitch_fov if pitch_fov is None else pitch_fov
        self._out_fov = np.array([yaw_fov, pitch_fov], dtype=np.float32)
        self._out_fov_rad = self._out_fov * np.pi / 180.0

        ### Culling occluded LiDAR
        # Create a list of offset coordinates within a radius R of the origin,
        # but excluding the origin itself.
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
        # print(self.device)

        cull_axis = torch.arange(-culling_r, culling_r + 1)
        offsets = torch.meshgrid(cull_axis, cull_axis, indexing="ij")
        offsets = torch.reshape(torch.stack(offsets, axis=-1), (-1, 2))  # (Nx2)
        offsets = offsets[torch.any(offsets != 0, axis=1)]  # remove origin
        offsets = offsets.to(self.device)
        self.offsets = offsets[None].type(torch.int32)  # expand_dims and cast

        ### Rendering masks and neural network model for sparse -> dense
        try:  # can only work with python 3.9
            rsrc_path = pkg_resources.files(resources)
        except AttributeError:
            with pkg_resources.path(vista, "resources") as p:
                rsrc_path = p
        # self.avg_mask = np.load(str(rsrc_path / "Lidar/avg_mask2.npy"))
        # self.avg_mask_pt = torch.tensor(self.avg_mask).to(self.device)

        self.load_model = load_model
        self.render_model = None
        path = rsrc_path / "Lidar/LidarFillerDevens.pt"
        if path.is_file() and load_model:
            logging.debug(f"Loading Lidar model from {path}")

            state_dict = torch.load(path, map_location=self.device)
            self.render_model = LidarModel(
                layers=int(state_dict["layers"]), filters=int(state_dict["filters"])
            )
            self.render_model.load_state_dict(state_dict)
            self.render_model.to(self.device)
            self.render_model.eval()

    def synthesize(
        self,
        trans: np.ndarray,
        rot: np.ndarray,
        pcd: Pointcloud,
        densification=False,
    ) -> Tuple[Pointcloud, np.ndarray]:
        """Apply rigid transformation to a dense pointcloud and return new
        dense representation or sparse pointcloud.

        Args:
            trans (np.ndarray): Translation vector.
            rot (np.ndarray): Rotation matrix.
            pcd (Pointcloud): Point cloud.

        Returns:
            Returns a tuple (``pointcloud``, ``array``), where ``pointcloud``
            is the synthesized point cloud with view point change from the
            transform (``trans``, ``rot``), and ``array`` is the dense depth
            map in 2D image space.

        """
        # Rigid transform of points
        R = transform.rot2mat(rot)
        pcd = pcd.transform(R, trans) # These are the points from the.las file cut by range only
        
        # Convert from new pointcloud to dense image with three channels
        # This should be where the circle appears
        visible = self._pcd2sparse(
            pcd,
            channels=(Point.DEPTH, Point.INTENSITY, Point.MASK),
            return_as_tensor=True,
            near=True,
        )
        
        
        # visible represents the points (with their repsective channels) before culling
        # print(torch.nonzero(~torch.isnan(visible[:,:,0].flatten())).shape)

        # occluded = self._pcd2sparse(
        #     pcd,
        #     channels=(Point.DEPTH, Point.INTENSITY, Point.MASK),
        #     return_as_tensor=True,
        #     near = False,
        # )
        # occluded[occluded - visible <= 0] = np.nan
        
        # Find occlusions and cull them from the rendering 
        occlusions, _ = self._cull_occlusions(visible[:, :, 0])
        visible[occlusions[:, 0], occlusions[:, 1]] = np.nan
        # occluded[~occlusions[:, 0], ~occlusions[:, 1]] = np.nan

        for idx, pcd in enumerate(
            [
                visible,
                # occluded,
            ]
        ):
            # Take depth and intensity channels of the point cloud
            intensity = pcd[:, :, 1]
            pcd = pcd[:, :, 0]
            
            pcd_idx = torch.nonzero(~torch.isnan(pcd)) # Get indices of non-nan values; those are the indices of the non occluded points
            
            # Remove occluded points (nan values)
            intensities = intensity.flatten()[~torch.isnan(intensity.flatten())]
            depths = pcd.flatten()[~torch.isnan(pcd.flatten())]
            
            fov_range = self._fov_rad[:, [1]] - self._fov_rad[:, [0]]
            step = fov_range / (self._dims)

            angles = torch.stack(
                (
                    torch.tensor(step[1]).to(self.device)
                    * (pcd_idx[:, 0] - int(self._dims[1, 0] / 2)),
                    torch.tensor(step[0]).to(self.device)
                    * (pcd_idx[:, 1] - int(self._dims[0, 0] / 2)),
                )
            ).T

            depths, indices = torch.sort(depths)
            angles = angles[indices, :]

            x = depths * torch.cos(angles[:, 0]) * torch.cos(angles[:, 1])
            y = depths * torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
            z = -depths * torch.sin(angles[:, 0])

            # These should be in radians already, I think.
            yaw = angles[:, 0]    # azimuth
            pitch = -angles[:, 1] # elevation
                                  # depths is given already

            '''This is where we will 'voxelize' our point cloud; commented out because this isn't being used
            # Spherically voxelize our point cloud
            # Divide spherical coordinates by the spherical voxel size define
            # by the sensor precision (manually defined for range)
            vx_yaw = torch.floor(yaw / self._res[0])
            vx_pch = torch.floor(pitch / self._res[1])
            vx_rng = torch.floor(depths / 0.1)

            # Get range for integration 
            yaw_low  = vx_yaw * self._res[0]
            pch_low  = vx_pch * self._res[1]
            rng_low  = vx_rng * 0.1
            yaw_high = yaw_low + self._res[0]
            pch_high = pch_low + self._res[1]
            rng_high = rng_low + 0.1
            
            volume = (1/3)*(
                torch.pow(rng_high, 3)-torch.pow(rng_low, 3)
                )*(
                torch.cos(pch_low)-torch.cos(pch_high)
                )*(
                yaw_high-yaw_low)

            max_volume = (1/3)*(
                torch.pow(SENSORCON_RANGE_HIGH, 3)-torch.pow(SENSORCON_RANGE_LOW, 3)
                )*(
                torch.cos(self._fov_rad[1, 0])-torch.cos(self._fov_rad[1, 1])
                )*(
                self._fov_rad[0, 1]-self._fov_rad[0, 0])

            azimuth_capacity = torch.floor((self._fov[0,1]-self._fov[0,0])/self._res[0])
            elevation_capacity = torch.floor((self._fov[1,1]-self._fov[1,0])/self._res[1])
            radius_capacity = torch.floor((SENSORCON_RANGE_HIGH-SENSORCON_RANGE_LOW)/0.1)
            total_voxels = azimuth_capacity * elevation_capacity * radius_capacity
            '''

            import math

            # Note that these are all of the points for each frame
            xyzypd = torch.stack((x, y, z, yaw, pitch, depths)).T
            # xyzi = torch.stack((x, y, z, intensities)).T


            import pandas

            df = pandas.DataFrame(xyzypd.cpu().numpy())
            df.columns = ["x", "y", "z", "yaw", "pitch", "depth"]
            # df.columns = ["x", "y", "z", "intensity"]
            if self._downsample:
                df = df.drop(df[df.depth < 50000].index)


            #outpath = f"./examples/vista_traces/lidar_output/output_{self._frame}_{self._res[0]: .2f}.txt"
            outfilepath = f"./examples/vista_traces/lidar_output/{self.roadsection_filename}_resolution={self._res[0]:.2f}"
            if not os.path.exists(outfilepath):
                os.makedirs(outfilepath)

            outpath = os.path.join(outfilepath, f"output_{self._frame}_{self._res[0]:.2f}.txt")

            outpath = "".join(outpath.split(" "))
            df.to_csv(
                outpath,
                index=False,
            )

        return visible, None

    def _pcd2sparse(
        self,
        pcd: Pointcloud,  
        channels: Point = Point.DEPTH,
        return_as_tensor: bool = False,
        near: bool = False,
    ) -> tensor_or_ndarray:
        """Convert from pointcloud to sparse image in polar coordinates.
        Fill image with specified features of the data (-1 = binary)."""

        if not isinstance(channels, list) and not isinstance(channels, tuple):
            channels = [channels]

        # Compute the values to fill and the indicies where to fill
        values = [pcd.get(channel) for channel in channels]
        values = np.stack(values, axis=1)
        inds = self._compute_sparse_inds(pcd) # Calculates where to put the channel (still before culling)

        # Re-order to fill points with smallest depth last
        if near:
            order = np.argsort(pcd.dist)[::-1] # Shorter ring at center
        else:
            order = np.argsort(pcd.dist)[::1]  # Larger ring at center
        values = values[order]
        inds = inds[:, order]

        # Creat the image and fill it
        img = np.empty((self._dims[1, 0], self._dims[0, 0], len(channels)), np.float32)
        img.fill(np.nan)
        img[-inds[1], inds[0], :] = values
        img[0, :] = np.nan #FIXME Top row of the sparse image is causing the ring issue; boundary problem?
        return torch.tensor(img).to(self.device) if return_as_tensor else img

    def _cull_occlusions(
        self,
        sparse: tensor_or_ndarray,
        depth_slack: float = 0.1,
    ) -> tensor_or_ndarray:

        # Coordinates where we have depth samples
        coords = torch.stack(torch.where(sparse > 0)).T

        # At each location, also compute coordinate for all of its neighbors
        samples = coords[:, None, :] + self.offsets  # (N, M, 2);  
        # print(samples)
        # print(samples.shape)
        # Does the ring artifact come from the bottom?

        # Collect the samples in each neighborhood
        samples[:, :, 0] = torch.clip(samples[:, :, 0], 0, sparse.shape[0] - 1)
        samples[:, :, 1] = torch.clip(samples[:, :, 1], 0, sparse.shape[1] - 1)
        neighbor_depth = sparse[samples[:, :, 0], samples[:, :, 1]]

        # For each location, compute the average depth of all neighbors
        valid = ~torch.isnan(neighbor_depth)
        scalar_zero = torch.zeros(1, 1).to(self.device)
        neighbor_depth = torch.where(valid, neighbor_depth, scalar_zero)
        avg_depth = torch.sum(neighbor_depth, axis=1) / torch.sum(
            valid.to(torch.float), axis=1
        )
        # median_depth = torch.median(neighbor_depth, axis=1).values 
        # NOTE I used .values here because at line 321 there is a type mismatch 
        # (median_depth: torch.return_types.median vs. Tensor)
        
        # Estimate if the location is occluded by measuring if its depth
        # greater than its surroundings (i.e. if it is behind its surroundings)
        # Some amound of slack can be added here to allow for edge cases.
        my_depth = sparse[coords[:, 0], coords[:, 1]]
        occluded = (my_depth - depth_slack) > avg_depth
        # occluded = (my_depth - depth_slack) > median_depth

        # Return the coordinates in the depth image which are occluded and
        # should be disregarded
        occluded_coords = coords[occluded]
        visible_coords = coords[~occluded]

        return occluded_coords, visible_coords

    def _cull_occlusions_np(
        self, sparse: np.ndarray, depth_slack: float = 0.1
    ) -> np.ndarray:

        coords = np.array(np.where(sparse > 0)).T

        samples = np.expand_dims(coords, 1) + self.offsets.numpy()
        samples[:, :, 0] = np.clip(samples[:, :, 0], 0, sparse.shape[0] - 1)
        samples[:, :, 1] = np.clip(samples[:, :, 1], 0, sparse.shape[1] - 1)

        neighbor_depth = sparse[samples[:, :, 0], samples[:, :, 1]]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_depth = np.nanmean(neighbor_depth, axis=1)

        my_depth = sparse[coords[:, 0], coords[:, 1]]

        # point is valid if it is closer than the average depth around it
        occluded = (my_depth - depth_slack) > avg_depth
        occluded_coords = coords[occluded]
        visible_coords = coords[~occluded]

        # remove (cull) all invalid points
        # sparse[occluded_coords[:, 0], occluded_coords[:, 1]] = np.nan
        # return sparse
        return occluded_coords, visible_coords

    def _sparse2dense(
        self, sparse: tensor_or_ndarray, method: str = "linear"
    ) -> tensor_or_ndarray:
        """Convert from sparse image representation of pointcloud to dense."""

        if method == "nn":
            sparse[torch.isnan(sparse)] = 0.0
            sparse = torch.nn.functional.pad(sparse, (0, 0, 4, 4, 4, 4))
            sparse = sparse[None].permute(0, 3, 1, 2)  # (1, 3, H+p, W+p)
            dense = self.render_model(sparse).detach()  # (1, 2, H+p, W+p)
            dense = dense.permute((0, 2, 3, 1))  # (1, H+p, W+p, 2)
            dense = dense[0, 4:-4, 4:-4]  # (H, W, 2)

        else:
            # mask all invalid values
            zs = np.ma.masked_invalid(sparse)

            # integer arrays for indexing
            grid_x, grid_y = np.meshgrid(
                np.arange(0, self._dims[0]), np.arange(0, self._dims[1])
            )

            # retrieve the valid, non-Nan, defined values
            valid_xs = grid_x[~zs.mask]
            valid_ys = grid_y[~zs.mask]
            valid_zs = zs[~zs.mask]

            # generate interpolated array of values
            dense = interpolate.griddata(
                (valid_xs, valid_ys), valid_zs, tuple((grid_x, grid_y)), method=method
            )
            dense[np.isnan(dense)] = 0.0
        return dense

    def _dense2pcd(self, dense: tensor_or_ndarray):
        """Sample mask from network and render points from mask"""
        # TODO: load trained masking network and feed dense through
        # For now, simply load a mask prior from training data and sample
        mask_shape = self.avg_mask.shape
        if isinstance(dense, torch.Tensor):
            mask = self.avg_mask_pt > torch.rand(size=mask_shape).to(self.device)
            pitch, yaw = torch.where(mask)
        else:
            mask = self.avg_mask > np.random.uniform(size=mask_shape)
            pitch, yaw = np.where(mask)

        pitch, yaw = self._coords2angles(pitch, yaw)
        rays = self._angles2rays(pitch, yaw)

        sampled_depth_and_ints = dense[mask]
        dist = sampled_depth_and_ints[:, 0]

        intensity = None
        if dense.shape[-1] == 2:  # intensity dimension
            intensity = sampled_depth_and_ints[:, 1]

        points = (dist * rays).T
        pcd = Pointcloud(points, intensity)
        return pcd

    def _coords2angles(
        self, pitch_coords: tensor_or_ndarray, yaw_coords: tensor_or_ndarray
    ) -> Tuple[tensor_or_ndarray, tensor_or_ndarray]:

        yaw = (
            yaw_coords * (self._fov_rad[0, 1] - self._fov_rad[0, 0]) / self._dims[0, 0]
            + self._fov_rad[0, 0]
        )
        pitch = (
            pitch_coords
            * (self._fov_rad[1, 0] - self._fov_rad[1, 1])
            / self._dims[1, 0]
            + self._fov_rad[1, 1]
        )
        return pitch, yaw

    def _angles2rays(
        self, pitch: tensor_or_ndarray, yaw: tensor_or_ndarray
    ) -> tensor_or_ndarray:

        with_torch = isinstance(pitch, torch.Tensor)
        cos = torch.cos if with_torch else np.cos
        sin = torch.sin if with_torch else np.sin
        stack = torch.stack if with_torch else np.array

        xyLen = cos(pitch)
        rays = stack([xyLen * cos(yaw), xyLen * sin(yaw), sin(pitch)])
        return rays

    def _compute_sparse_inds(self, pcd: Pointcloud) -> np.ndarray:
        """Compute the indicies on the image representation which will be
        filled for a given pointcloud"""

        # project point cloud to 2D point map
        yaw = np.arctan2(pcd.y, pcd.x)
        pitch = np.arcsin(pcd.z / pcd.dist)
        angles = np.stack((yaw, pitch))

        fov_range = self._fov_rad[:, [1]] - self._fov_rad[:, [0]] # total coverage of yaw, pitch in radians
        
        slope = self._dims / fov_range
        inds = slope * (angles - self._fov_rad[:, [0]])

        inds = np.floor(inds).astype(int)

        np.clip(inds, np.zeros((2, 1)), self._dims - 1, out=inds)

        return inds
