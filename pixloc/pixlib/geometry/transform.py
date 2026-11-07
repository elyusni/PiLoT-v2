import pyproj
import numpy as np
# from render2loc_tan import Render2Loc

### SETTING EXP PLACE (TODO:不同实验地点影响get_CRS和json文件选择)
PLACE_CHANGSHA = 0

def get_rotation_enu_in_ecef(lon, lat):
    """
    @param: lon, lat Longitude and latitude in degree
    @return: 3x3 rotation matrix of heading-pith-roll ENU in ECEF coordinate system
    Reference: https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf, Section 4.3, 4.1
    Reference: https://www.fossen.biz/wiley/ed2/Ch2.pdf, p29
    """
    # 将角度转换为弧度
    latitude_rad = np.radians(lat)
    longitude_rad = np.radians(lon)

    # 计算向上的向量（Up Vector）
    up = np.array([
        np.cos(longitude_rad) * np.cos(latitude_rad),
        np.sin(longitude_rad) * np.cos(latitude_rad),
        np.sin(latitude_rad)
    ])

    # 计算向东的向量（East Vector）
    east = np.array([
        -np.sin(longitude_rad),
        np.cos(longitude_rad),
        0
    ])

    # 计算向北的向量（North Vector），即up向量和east向量的外积（叉积）
    north = np.cross(up, east)

    # 构建局部到世界坐标系的转换矩阵
    local_to_world = np.zeros((3, 3))
    local_to_world[:, 0] = east  # 东向分量
    local_to_world[:, 1] = north  # 北向分量
    local_to_world[:, 2] = up  # 上向分量
    return local_to_world

# ECEF (EPSG:4978) -> WGS84经纬高 (EPSG:4326)
_transformer_ecef2wgs84 = pyproj.Transformer.from_crs(
    "EPSG:4978",  # ECEF
    "EPSG:4326",  # lon, lat, h (degrees, meters)
    always_xy=True,
)
def ECEF_to_WGS84(pos):
    """
    pos: shape (3,) or (N,3) array-like of [X, Y, Z] in meters (ECEF).
    return:
        if input (3,):  (lon, lat, h)
        if input (N,3): array of shape (N,3) with columns [lon, lat, h]
    """
    pos = np.asarray(pos, dtype=float)

    # 单点
    if pos.ndim == 1 and pos.shape[0] == 3:
        x, y, z = pos
        lon, lat, h = _transformer_ecef2wgs84.transform(x, y, z)
        return lon, lat, h

    # 批量 (N,3)
    assert pos.ndim == 2 and pos.shape[1] == 3, "pos must be (3,) or (N,3)"
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    lon, lat, h = _transformer_ecef2wgs84.transform(x, y, z)
    return np.stack([lon, lat, h], axis=1)

def WGS84_to_ECEF(pos):
    lon, lat, height = pos
    transprojr = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        always_xy=True,
    )
    xpjr, ypjr, zpjr = transprojr.transform(lon, lat, height, radians=False)
    return [xpjr, ypjr, zpjr]

# def wgs84tocgcs2000(trans, exp_place):
#     """Convert coordinates from WGS84 to CGCS2000.

#     Args:
#         trans (list): [lon, lat, height] in WGS84 format
#     """
#     lon, lat, height = trans  # Unpack the WGS84 coordinates

#     # wgs84, cgcs2000 = get_CRS(exp_place)
#     wgs84

#     # Create a transformer from WGS84 to CGCS2000
#     transformer = pyproj.Transformer.from_crs(wgs84, cgcs2000, always_xy=True)

#     # Perform the transformation
#     x, y = transformer.transform(lon, lat)

#     # Return the transformed coordinates as a list
#     return [x, y, height]  # Keep the original height from WGS84

#!wxyz
#!wxyz
