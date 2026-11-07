import numpy as np
from scipy.spatial.transform import Rotation as R
import pyproj

_WGS84_CRS = pyproj.CRS('EPSG:4326')
_CGCS2000_CRS = pyproj.CRS('EPSG:4547')
_WGS84_TO_CGCS2000 = pyproj.Transformer.from_crs(
    _WGS84_CRS, _CGCS2000_CRS, always_xy=True
)
_DEFAULT_INTRINSICS = np.array([
    [1350.0, 0.0, 957.85],
    [0.0, 1350.0, 537.55],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
_DEFAULT_Q_INTRINSICS_INFO = [1920, 1080, 1920, 1080, 24.00]

def convert_euler_to_matrix(euler_xyz):

    ret = R.from_euler('xyz', [euler_xyz[1]-90, float(euler_xyz[2]), -euler_xyz[0]], degrees=True)

    rotation_matrix = ret.as_matrix()

    return rotation_matrix

def wgs84tocgcs2000(trans):
    lon, lat, height = trans

    x, y = _WGS84_TO_CGCS2000.transform(lon, lat)

    return [x, y, height]

def transform_colmap_pose_intrinsic(pose_data):

    lon, lat, alt, roll, pitch, yaw = pose_data
    yaw = -yaw
    pitch = pitch - 90

    euler_xyz = [yaw, pitch, roll]
    r_c2w = convert_euler_to_matrix(euler_xyz)

    trans = [lon, lat, alt]
    t_c2w = wgs84tocgcs2000(trans)

    T = np.identity(4)
    T[:3, :3] = r_c2w.T
    T[:3, 3] = -r_c2w.T @ t_c2w
    poses_dict= T

    intrinsics_dict = _DEFAULT_INTRINSICS

    osg_dict = euler_xyz
    q_intrinsics_info = _DEFAULT_Q_INTRINSICS_INFO

    return poses_dict, intrinsics_dict, q_intrinsics_info, osg_dict
