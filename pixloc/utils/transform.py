import torch
import pyproj
import numpy as np
from pykalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
# from render2loc_tan import Render2Loc

### SETTING EXP PLACE (TODO:不同实验地点影响get_CRS和json文件选择)
PLACE_CHANGSHA = 0
def move_inputs_to_cuda(*args, device=None):
        """
        递归地将输入参数中的所有 Tensor、Pose、Camera（如果定义了 .to() 方法）转到 CUDA。
        返回一个 tuple，顺序与 args 保持一致。
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def move(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            elif hasattr(x, 'to') and callable(getattr(x, 'to')):  # 支持 Pose、Camera 等
                return x.to(device)
            elif isinstance(x, (list, tuple)):
                return type(x)(move(i) for i in x)
            elif isinstance(x, dict):
                return {k: move(v) for k, v in x.items()}
            else:
                return x

        return tuple(move(arg) for arg in args)
def euler_angles_to_matrix_ECEF(euler_angles, trans):
    lon, lat, _ = trans
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    # q_w2c = rotmat2qvec(R_c2w.transpose())  # return wxyz (colmap pnp return xyzw)
    t_c2w = WGS84_to_ECEF(trans)
    # R_w2c_in_ecef = R_c2w.transpose() # 和enu的差异是第二行和第三行取负号
    # t_w2c = -R_w2c_in_ecef.dot(t_c2w)

    T_render_in_ECEF_c2w = np.eye(4)
    T_render_in_ECEF_c2w[:3, :3] = R_c2w
    T_render_in_ECEF_c2w[:3, 3] = t_c2w
    return T_render_in_ECEF_c2w

def pixloc_to_osg(T_refined_c2w):
    R_c2w, t_c2w = T_refined_c2w[:3, :3], T_refined_c2w[:3, 3]
    t_c2w_wgs84 = ECEF_to_WGS84(t_c2w)
    lon, lat, _ = t_c2w_wgs84
    # 计算从ENU到ECEF的旋转矩阵
    rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    # 将ECEF姿态矩阵转换为ENU姿态矩阵
    rot_ecef_in_enu = rot_ned_in_ecef.T  #! ECEF TO WGS84 transformation
    # 将ECEF姿态矩阵转换为ENU姿态矩阵
    rot_pose_in_enu = np.matmul(rot_ecef_in_enu, R_c2w)
    # 从ENU姿态矩阵中提取欧拉角
    rot_pose_in_enu_obj = R.from_matrix(rot_pose_in_enu)
    euler_angles_in_enu = rot_pose_in_enu_obj.as_euler('xyz', degrees=True)

    R_w2c = R_c2w.T
    t_w2c = np.array(-R_w2c.dot(t_c2w))

    T_ECEF = np.concatenate((R_w2c, np.array([t_w2c]).transpose()), axis=1)

    # R_w2c = T_ECEF[:3, :3]
    # t_w2c = T_ECEF[:3, 3]
    # R_c2ww = R_w2c.T
    # t_c2ww = np.array(-R_c2ww.dot(t_w2c))
    # t_c2w_wgs84 = ECEF_to_WGS84(t_c2ww)
    # lon, lat, _ = t_c2w_wgs84
    # # 计算从ENU到ECEF的旋转矩阵
    # rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    # # 将ECEF姿态矩阵转换为ENU姿态矩阵
    # rot_ecef_in_enu = rot_ned_in_ecef.T  #! ECEF TO WGS84 transformation
    # # 将ECEF姿态矩阵转换为ENU姿态矩阵
    # rot_pose_in_enu = np.matmul(rot_ecef_in_enu, R_c2ww)
    # # 从ENU姿态矩阵中提取欧拉角
    # rot_pose_in_enu_obj = R.from_matrix(rot_pose_in_enu)
    # euler_angles_in_enu = rot_pose_in_enu_obj.as_euler('xyz', degrees=True)

    # R_c2w = R_w2c.T
    # t_c2w = np.array(-R_c2w.dot(t_w2c))
    # kf_current_frame_es_pose = t_c2ww+euler_angles_in_enu
    kf_current_frame_es_pose = np.concatenate((t_c2w, euler_angles_in_enu))
    return euler_angles_in_enu, t_c2w_wgs84, T_ECEF, kf_current_frame_es_pose
def kf_predictor(observations, num_candidates = 128):
    # 初始化卡尔曼滤波器
    # 计算变化量 Δp 和 Δr
    observations = np.array(observations)
    assert observations.shape[1] == 6, "Expected input shape: (N, 6)"

    dt = 1.0  # 默认时间步长

    # 状态：位置+欧拉角+线速度+角速度 共12维
    # [x, y, z, pitch, roll, yaw, vx, vy, vz, v_pitch, v_roll, v_yaw]
    A = np.eye(12)
    for i in range(6):
        A[i, i + 6] = dt  # 添加速度分量影响

    H = np.hstack([np.eye(6), np.zeros((6, 6))])  # 观测只包含位置和姿态

    kf = KalmanFilter(
        transition_matrices=A,
        observation_matrices=H,
        initial_state_mean=np.zeros(12),
        initial_state_covariance=np.eye(12) * 0.1,
        transition_covariance=np.eye(12) * 0.05,
        observation_covariance=np.eye(6) * 0.1
    )

    # 初始化
    state_mean = np.zeros(12)
    state_mean[:6] = observations[0]
    state_cov = np.eye(12) * 0.1

    filtered_state_means = []
    filtered_state_covs = []
    for obs in observations:
        state_mean, state_cov = kf.filter_update(state_mean, state_cov, observation=obs)
        filtered_state_means.append(state_mean)
        filtered_state_covs.append(state_cov)
    # 取最后一步进行预测
    last_mean = filtered_state_means[-1]
    # last_cov = filtered_state_covs[-1]
    # 从预测分布中生成 100 个候选位姿

    # 预测下一时刻的状态
    next_state_mean = A @ last_mean
    # next_state_cov = A @ last_cov @ A.T + kf.transition_covariance

    # 从预测分布采样候选位姿
    # candidates = np.random.multivariate_normal(next_state_mean, next_state_cov, num_candidates)[:, :6]
    most_likely_pose = next_state_mean[:6]  # [x, y, z, pitch, roll, yaw]

    euler = most_likely_pose[3:]
    t_c2w_ecef = most_likely_pose[:3]
    # rot_pose_in_enu = R.from_euler('xyz', euler, degrees=True).as_matrix()
    t_c2w_wgs84 = ECEF_to_WGS84(t_c2w_ecef)
    # lon, lat, _ = t_c2w_wgs84

    # new_origin = t_c2w_ecef

    # rot_enu_to_ecef = get_rotation_enu_in_ecef(lon, lat)  # 假设函数返回 (3, 3) 矩阵
    # R_c2w = np.matmul(rot_enu_to_ecef, rot_pose_in_enu)

    # most_likely_pose_T_w2c = np.eye(4)
    # most_likely_pose_T_w2c[:3, :3] = R_c2w.T
    # most_likely_pose_T_w2c[:3, 3] = -R_c2w.T @ t_c2w_ecef

    # T_next_frame = Pose.from_Rt(most_likely_pose_T_w2c[:3, :3], most_likely_pose_T_w2c[:3, 3])  # w2c

    # 候选位姿构建
    # candidate_delta_poses = candidates - observations[-1, :]
    # translations = candidate_delta_poses[:, :3]
    # euler_angles = candidate_delta_poses[:, 3:]

    # 欧拉角转旋转向量
    # rotations = R.from_euler('xyz', euler_angles, degrees=False)
    # rotation_vectors = rotations.as_rotvec()

    # 构建 torch Tensor
    # translations_tensor = torch.tensor(translations, dtype=torch.float32)
    # directions = F.normalize(translations_tensor, dim=-1)
    # translation_lengths = torch.norm(translations_tensor, dim=-1, keepdim=True)
    # random_t = directions * translation_lengths

    # random_aa = torch.tensor(rotation_vectors, dtype=torch.float32)
    # random_pose = Pose.from_aa(random_aa, random_t)

    # 更新观测队列
    observations = np.vstack([observations, most_likely_pose])
    observations = observations[1:, :]
    filtered_state_means = np.array(filtered_state_means)  # (N, 12)
    # plot_kf_trajectory_save(filtered_state_means, most_likely_pose)
    # visualize_prediction(filtered_state_means, most_likely_pose, candidates)

    return observations, euler, t_c2w_wgs84

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

def ECEF_to_WGS84(pos):
    xpjr, ypjr, zpjr = pos
    transprojr = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        "EPSG:4326",
        always_xy=True,
    )
    lon, lat, height = transprojr.transform(xpjr, ypjr, zpjr, radians=False)
    return [lon, lat, height]
def WGS84_to_ECEF_tensor(pos, device='cuda', dtype=torch.float32):
    """
    Convert WGS84 (lon, lat, height) → ECEF (x, y, z) using PyTorch on GPU.

    Args:
        pos: Iterable of 3 elements or Tensor of shape (..., 3),
             where last dim = [lon_deg, lat_deg, height_m].
        device: 'cuda' or 'cpu'
        dtype: torch.float32 or torch.float64

    Returns:
        Tensor of shape (..., 3) = [X, Y, Z] in meters
    """
    # WGS84 constants
    a = torch.tensor(6378137.0, device=device, dtype=dtype)          # semi-major axis
    f = torch.tensor(1 / 298.257223563, device=device, dtype=dtype)  # flattening
    e2 = f * (2 - f)                                                  # first eccentricity squared

    # Pack input into tensor
    p = torch.as_tensor(pos, device=device, dtype=dtype)
    lon_deg = p[..., 0]
    lat_deg = p[..., 1]
    h       = p[..., 2]

    # to radians
    lon = lon_deg * (torch.pi / 180.0)
    lat = lat_deg * (torch.pi / 180.0)

    sin_lat = torch.sin(lat)
    cos_lat = torch.cos(lat)
    cos_lon = torch.cos(lon)
    sin_lon = torch.sin(lon)

    # Radius of curvature in the prime vertical
    N = a / torch.sqrt(1 - e2 * sin_lat * sin_lat)  # shape (...)

    # Compute ECEF
    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (N * (1 - e2) + h) * sin_lat

    return torch.stack([X, Y, Z], dim=-1)
def WGS84_to_ECEF(pos):
    lon, lat, height = pos
    transprojr = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        always_xy=True,
    )
    xpjr, ypjr, zpjr = transprojr.transform(lon, lat, height, radians=False)
    return [xpjr, ypjr, zpjr]

#!wxyz
#!wxyz
