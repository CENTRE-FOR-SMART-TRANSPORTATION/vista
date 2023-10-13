import numpy as np
import open3d as o3d
class LasPointCloud:
    """
    Container class for the .las file. Cuts down on unused fields from the
    raw .las file itself.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        gps_time: np.ndarray,
        scan_angle_rank: np.ndarray,
        point_source_ID: np.ndarray,
        intensity: np.ndarray,
        lasfilename: str,
    ):
        self.__x = x
        self.__y = y
        self.__z = z
        self.__gps_time = gps_time
        self.__scan_angle_rank = scan_angle_rank
        self.__point_source_ID = point_source_ID
        self.__intensity = intensity
        self.__lasfilename = lasfilename

    # Getters
    def getX(self) -> np.ndarray:
        return self.__x

    def getY(self) -> np.ndarray:
        return self.__y

    def getZ(self) -> np.ndarray:
        return self.__z

    def getGPSTime(self) -> np.ndarray:
        return self.__gps_time

    def getScanAngleRank(self) -> np.ndarray:
        return self.__scan_angle_rank

    def getPointSourceID(self) -> np.ndarray:
        return self.__point_source_ID

    def getIntensity(self) -> np.ndarray:
        return self.__intensity

    def getLasFileName(self) -> str:
        return self.__lasfilename

    # Setters (just in case if we want to work with future .las clouds, but these probably shouldn't be used)
    def setX(self, x: np.ndarray) -> None:
        self.__x = x

    def setY(self, y: np.ndarray) -> None:
        self.__y = y

    def setZ(self, z: np.ndarray) -> None:
        self.__z = z

    def setGPSTime(self, gps_time: np.ndarray) -> None:
        self.__gps_time = gps_time

    def setScanAngleRank(self, scan_angle_rank: np.ndarray) -> None:
        self.__scan_angle_rank = scan_angle_rank

    def setPointSourceID(self, point_source_id: np.ndarray) -> None:
        self.__point_source_ID = point_source_id

    def setIntensity(self, intensity: np.ndarray) -> None:
        self.__intensity = intensity

    def setLasFileName(self, lasfilename: str) -> None:
        self.__lasfilename = lasfilename


class Trajectory:
    """Container class for the trajectory"""

    def __init__(
        self,
        observer_points: np.ndarray,
        road_points: np.ndarray,
        forwards: np.ndarray,
        leftwards: np.ndarray,
        upwards: np.ndarray,
    ) -> None:
        self.__observer_points = observer_points
        self.__road_points = road_points
        self.__forwards = forwards
        self.__leftwards = leftwards
        self.__upwards = upwards

        pass

    # Getters
    def getObserverPoints(self) -> np.ndarray:
        return self.__observer_points

    def getRoadPoints(self) -> np.ndarray:
        return self.__road_points

    def getForwards(self) -> np.ndarray:
        return self.__forwards

    def getLeftwards(self) -> np.ndarray:
        return self.__leftwards

    def getUpwards(self) -> np.ndarray:
        return self.__upwards

    def getNumPoints(self) -> np.int32:
        return self.__road_points.shape[0]

    # Setters (just in case if we want to work with future trajectories)
    def setObserverPoints(self, observer_points: np.ndarray) -> None:
        self.__observer_points = observer_points

    def setRoadPoints(self, road_points: np.ndarray) -> None:
        self.__road_points = road_points

    def setForwards(self, forwards: np.ndarray) -> None:
        self.__forwards = forwards

    def setLeftwards(self, leftwards: np.ndarray) -> None:
        self.__leftwards = leftwards

    def setUpwards(self, upwards: np.ndarray) -> None:
        self.__upwards = upwards


class SensorConfig:
    '''
    Container class for the sensor configuration.
    '''

    def __init__(self, numberSensors, horizAngRes, verticAngRes, e_low, e_high, a_low, a_high, r_low, r_high):
        self.numberSensors = numberSensors
        self.horizAngRes = horizAngRes
        self.verticAngRes = verticAngRes
        self.e_low = e_low
        self.e_high = e_high
        self.a_low = a_low
        self.a_high = a_high
        self.r_low = r_low
        self.r_high = r_high
    pass

    sensor_config_filename = None

    # We shouldn't need setters, let alone getters since we are
    # creating only one container object, but I did it just in case.
    def getNumberSensors(self):
        return self.numberSensors

    def getHorizAngRes(self):
        return self.horizAngRes

    def getVerticAngRes(self):
        return self.verticAngRes

    def getELow(self):
        return self.e_low

    def getEHigh(self):
        return self.e_high

    def getALow(self):
        return self.a_low

    def getAHigh(self):
        return self.a_high

    def getRLow(self):
        return self.r_low

    def getRHigh(self):
        return self.r_high

class PointCloudOpener:
    # Opens one specified point cloud as a Open3D tensor point cloud for parallelism
    def open_point_cloud(
        self, path_to_scenes: str, frame: int, res: np.float32
    ) -> o3d.t.geometry.PointCloud:
        """Reads a specified point cloud from a path into memory.
        This is called in the parallelized loop in obtain_scenes().

        Args:
            path2scenes (str): The path to the folder containing the scenes.
            frame (int): The frame of the particular scene.
            res (np.float32): The resolution of the sensor at which the scene was recorded.
            This should be given in the filename, where scene names are guaranteed to be
            "output_<FRAME>_<RES>.txt".

        Returns:
            pcd (o3d.t.geometry.PointCloud): Our point cloud, in tensor format.
        """

        scene_name = f"output_{frame}_{res:.2f}.txt"
        path_to_scene = os.path.join(path_to_scenes, scene_name)
        # print(f"Opening {scene_name} as pcd...")

        # Skip our header, and read only XYZ coordinates
        df = pd.read_csv(path_to_scene, skiprows=0, usecols=[0, 1, 2, 3])
        xyz = df.iloc[:, :3].to_numpy() / 1000  # Extract XYZ coordinates
        # Extract intensity values
        intensity = df.iloc[:, 3].to_numpy() / 1000

        # Create Open3D point cloud object with tensor values.
        # For parallelization, outputs must be able to be serialized
        pcd = o3d.t.geometry.PointCloud(o3d.core.Device("CPU:0"))
        pcd.point.positions = o3d.core.Tensor(
            xyz, o3d.core.float32, o3d.core.Device("CPU:0")
        )

        # Set the colors of the point cloud using the intensity-based color map
        pcd.point.colors = o3d.core.Tensor(
            intensity, o3d.core.float32, o3d.core.Device("CPU:0"))

        return pcd