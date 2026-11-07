import torch
import numpy as np
import pyproj
import copy
import math
from pixloc.utils.transform import WGS84_to_ECEF_tensor
from scipy.spatial.transform import Rotation as R
from ..pixlib.geometry import Pose, Camera
import torch.nn.functional as F

_TRANSFORMER_CACHE = {}

def _get_cached_transformer(src_crs, dst_crs):
    key = (src_crs, dst_crs)
    transformer = _TRANSFORMER_CACHE.get(key)
    if transformer is None:
        transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        _TRANSFORMER_CACHE[key] = transformer
    return transformer

def interpolate_depth_grid(pos, depth):
    """
    pos: Tensor[N,2], 每行是 (i, j) 像素坐标
    depth: Tensor[H,W]
    返回: depth_interp[valid], pos_valid.T, ids_valid
    """
    # 1) 准备
    device = depth.device
    H, W = depth.shape
    # 把 depth 变为 [1,1,H,W]
    depth4 = depth.unsqueeze(0).unsqueeze(0)

    # 2) 归一化到 [-1,1]，注意 grid_sample 的 coord 是 (x=j, y=i)
    j = pos[:,1].to(device)
    i = pos[:,0].to(device)
    # x 栏：2*j/(W-1)-1;  y 栏：2*i/(H-1)-1
    x = 2.0*j/(W-1) - 1.0
    y = 2.0*i/(H-1) - 1.0
    grid = torch.stack([x, y], dim=1)        # [N,2]
    grid = grid.view(1, -1, 1, 2)            # [1, N,1,2]

    # 3) 调用 grid_sample
    sampled = F.grid_sample(
        depth4, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # [1,1,N,1]
    sampled = sampled.view(-1)               # [N]

    # 4) 掩码：坐标越界 or depth==0
    valid = (
        (i >= 0) & (i <= H-1) &
        (j >= 0) & (j <= W-1) &
        (sampled > 0)
    )
    ids   = valid.nonzero(as_tuple=False).view(-1)
    pos_v = pos[ids].t()                     # [2, M]
    depth_v = sampled[ids]                   # [M]

    return depth_v, pos_v, ids

def read_valid_depth(mkpts1r, depth=None, device = 'cuda'):
    depth = torch.tensor(depth).to(device)
    mkpts1r = mkpts1r.float().to(device)
    mkpts1r_inter = mkpts1r[:, [1, 0]].to(device)
    # depth_interpolated, _, valid = interpolate_depth(mkpts1r_inter, depth)

    # mkpts1r = mkpts1r.float().to(device)
    # mkpts1r_inter = mkpts1r[:, [1, 0]].to(device)
    depth_interpolated, _, valid = interpolate_depth_grid(mkpts1r_inter, depth)

    return depth_interpolated, valid
def generate_random_aa_and_t_cuda(
    n: int,
    min_offset_angle: float, max_offset_angle: float,
    min_offset_translation: float, max_offset_translation: float,
    device=torch.device("cuda"),
    dtype=torch.float32,
):
    # 1) 生成随机方向（单位向量）
    axis = torch.randn(n, 3, device=device, dtype=dtype)
    axis = axis / axis.norm(dim=1, keepdim=True)  # 归一化为单位向量

    direction = torch.randn(n, 3, device=device, dtype=dtype)
    direction = direction / direction.norm(dim=1, keepdim=True)

    # 2) 生成角度（单位为度，再转弧度）
    delta_angle = max_offset_angle - min_offset_angle
    angles_deg = min_offset_angle + delta_angle * torch.rand(n, device=device, dtype=dtype)
    angles_rad = angles_deg * (math.pi / 180.0)
    aa = axis * angles_rad.unsqueeze(1)  # (n, 3)

    # 3) 生成平移（单位：米）
    delta_trans = max_offset_translation - min_offset_translation
    trans_mag = min_offset_translation + delta_trans * torch.rand(n, device=device, dtype=dtype)
    trans = direction * trans_mag.unsqueeze(1)  # (n, 3)
    aa[0] = 0.0
    trans[0] = 0.0
    return aa, trans

def generate_rotvecs_cuda_sym3d(
    base_pitch, base_roll, base_yaw,
    max_pitch: float, pitch_step: float,
    max_yaw:   float,   yaw_step:   float,
    max_roll:  float,  roll_step:  float,
    device:   str = 'cuda'):
    """
    在 CUDA 上生成 (pitch, roll, yaw) 三维对称网格采样：
      - pitch 从 -max_pitch 到 +max_pitch，步长为 pitch_step
      - yaw   从 -max_yaw   到 +max_yaw，   步长为 yaw_step
      - roll  从 -max_roll  到 +max_roll，  步长为 roll_step

    如果某个 max = 0，则只在该维度采样 0；否则会在 [-max, +max] 区间内按照 step 生成序列，
    包含正负两侧（如果 (2*max) 不是 step 的整数倍，则右边最后一个采样点会略小于 +max）。

    返回张量形状为 [N, 3]，其中
      N = num_pitch * num_yaw * num_roll，
    每行即为一个 (pitch, roll, yaw) 三元组，dtype=float32 且在 CUDA 上。

    示例：
        # 在 [-10, +10] 以 5 度步长采样 pitch、yaw、roll
        rotvecs = generate_rotvecs_cuda_sym3d(
            max_pitch=10.0,  pitch_step=5.0,
            max_yaw=10.0,    yaw_step=5.0,
            max_roll=10.0,  roll_step=5.0,
            device='cuda'
        )
        # 此时 pitch_vals = [-10, -5, 0, 5, 10]，yaw_vals、roll_vals 同理
        # N = 5 * 5 * 5 = 125，所以 rotvecs.shape = [125, 3]
    """

    # ------- 1. 针对每个维度，生成 [-max, +max] 等步长值 -------
    def _symmetric_range(max_val: float, step: float) -> torch.Tensor:
        """
        生成一个对称采样向量：
          如果 max_val == 0: 返回 tensor([0.0])
          否则：采样 [-max_val, -max_val+step, ..., 0, ..., max_val-step, max_val]（包含正负两端）
        """
        if step <= 0 or max_val < 0:
            raise ValueError("step 必须 > 0 且 max_val 必须 >= 0")
        if max_val == 0:
            return torch.tensor([0.0], device=device, dtype=torch.float32)

        # torch.arange(start, end, step) 会生成 [start, start+step, ..., < end]
        # 我们希望包含 -max_val 和 +max_val 两侧，先从 0 到 +max_val 再映射
        pos = torch.arange(0.0, max_val + 1e-6, step, device=device, dtype=torch.float32)
        # 如果 pos 的最后一个元素略大于 max_val（due to float 误差），可以 clamp
        if pos[-1] > max_val:
            pos[-1] = max_val

        # 剔除第一个 0，生成正半区采样；0 在最中心单独出现一次
        if pos.shape[0] > 1:
            pos_nonzero = pos[1:]  # 跳过 0
            # 负半区 = 正半区 取反并倒序，以保持顺序从小到大
            neg = (-pos_nonzero).flip(0)
            # 合并负、中、正
            return torch.cat((neg, pos[:1], pos_nonzero), dim=0)
        else:
            # 只有一个元素 0
            return pos

    pitch_vals = _symmetric_range(max_pitch, pitch_step)
    yaw_vals   = _symmetric_range(max_yaw,   yaw_step)
    roll_vals  = _symmetric_range(max_roll,  roll_step)

    # ------- 2. 使用 meshgrid 生成三维网格 -------
    P, Y, R = torch.meshgrid(
        pitch_vals, yaw_vals, roll_vals, indexing='ij'
    )  # P.shape == (num_pitch, num_yaw, num_roll)

    # ------- 3. 展平至 (N,) 并拼成 (N,3) -------
    N = P.numel()
    P_flat = P.reshape(N)
    Y_flat = Y.reshape(N)
    R_flat = R.reshape(N)

    # P_flat[0] = 0
    # R_flat[0] = 0
    # Y_flat[0] = 0

    pitch = P_flat + base_pitch
    roll  = R_flat + base_roll
    yaw   = Y_flat + base_yaw
    # 拼成 (N,3)
    return torch.stack((pitch, roll, yaw), dim=1)       # float tensor on GPU
def get_rotation_enu_in_ecef_tensor(lon, lat, device='cuda', dtype=torch.float32):
    """
    Compute the 3×3 ENU→ECEF rotation matrix on GPU.

    Args:
        lon (float or Tensor): Longitude in degrees.
        lat (float or Tensor): Latitude in degrees.
        device (str or torch.device): e.g. 'cuda' or 'cpu'.
        dtype (torch.dtype): e.g. torch.float32.

    Returns:
        Tensor of shape (3,3) on specified device.
    """
    # 1) 转成 tensor
    lon_t = torch.as_tensor(lon, device=device, dtype=dtype)
    lat_t = torch.as_tensor(lat, device=device, dtype=dtype)

    # 2) 角度转弧度
    lon_rad = lon_t * (torch.pi / 180.0)
    lat_rad = lat_t * (torch.pi / 180.0)

    # 3) 计算 up, east, north
    up = torch.stack([
        torch.cos(lon_rad) * torch.cos(lat_rad),
        torch.sin(lon_rad) * torch.cos(lat_rad),
        torch.sin(lat_rad)
    ])

    east = torch.stack([
        -torch.sin(lon_rad),
         torch.cos(lon_rad),
         torch.zeros_like(lon_rad)
    ])

    north = torch.cross(up, east, dim=0)

    # 4) 拼矩阵，列分别是 east/north/up
    rot = torch.stack([east, north, up], dim=1)  # shape (3,3)
    return rot
def euler_to_rotm_batch(angles, translation, degrees=True, device='cuda'):
    # angles: (N,3) tensor representing [pitch, roll, yaw]
    if degrees:
        angles = angles * (torch.pi/180.0)
    p, r, y = angles.unbind(1)   # each (N,)

    # 生成各自的旋转矩阵 Rx(p), Ry(r), Rz(y)
    # Rx (around x axis = pitch)
    zero = torch.zeros_like(p); one = torch.ones_like(p)
    Rx = torch.stack([
        torch.stack([ one, zero,    zero], dim=1),
        torch.stack([ zero, torch.cos(p), -torch.sin(p)], dim=1),
        torch.stack([ zero, torch.sin(p),  torch.cos(p)], dim=1),
    ], dim=1)  # (N,3,3)

    # Ry (around y axis = roll)
    Ry = torch.stack([
        torch.stack([ torch.cos(r), zero, torch.sin(r)], dim=1),
        torch.stack([ zero,        one,  zero      ], dim=1),
        torch.stack([-torch.sin(r), zero, torch.cos(r)], dim=1),
    ], dim=1)

    # Rz (around z axis = yaw)
    Rz = torch.stack([
        torch.stack([torch.cos(y), -torch.sin(y), zero], dim=1),
        torch.stack([torch.sin(y),  torch.cos(y), zero], dim=1),
        torch.stack([zero,          zero,         one ], dim=1),
    ], dim=1)

    # 按顺序乘：R = ENU2ECEF @ Rz @ Rx @ Ry  （根据你的定义调整）
    # 假设 get_rotation_enu_in_ecef 返回 (3,3) CPU or GPU tensor:
    lon, lat, _ = translation
    rot_enu_in_ecef = get_rotation_enu_in_ecef_tensor(lon, lat).to(device)
    R_local = Rz @ Rx @ Ry                         # (N,3,3)
    R_batch = rot_enu_in_ecef.unsqueeze(0) @ R_local  # (N,3,3)
    return R_batch

def generate_translations_ecef_sym3d(
    base_translation_wgs84,  # [lon, lat, h] WGS84 坐标
    max_dx=0.0, dx_step=1.0,  # ECEF X 方向扰动 (米)
    max_dy=0.0, dy_step=1.0,  # ECEF Y 方向扰动 (米)
    max_dz=0.0, dz_step=1.0,  # ECEF Z 方向扰动 (米)
    device='cuda'
):
    """
    在 ECEF 坐标系下生成位置扰动网格

    Args:
        base_translation_wgs84: [lon, lat, h] WGS84 基准坐标
        max_dx, dx_step: ECEF X 方向最大扰动和步长 (米)
        max_dy, dy_step: ECEF Y 方向最大扰动和步长 (米)
        max_dz, dz_step: ECEF Z 方向最大扰动和步长 (米)
        device: 'cuda' or 'cpu'

    Returns:
        translations_ecef: [M, 3] ECEF 坐标
    """
    # 1. WGS84 -> ECEF (基准位置)
    base_ecef = WGS84_to_ECEF_tensor(base_translation_wgs84, device=device)

    # 2. 生成扰动值 (对称网格)
    def _symmetric_range(max_val, step):
        if max_val == 0 or step <= 0:
            return torch.tensor([0.0], device=device, dtype=torch.float32)
        pos = torch.arange(0.0, max_val + 1e-6, step, device=device, dtype=torch.float32)
        if pos[-1] > max_val:
            pos[-1] = max_val
        if pos.shape[0] > 1:
            pos_nonzero = pos[1:]
            neg = (-pos_nonzero).flip(0)
            return torch.cat((neg, pos[:1], pos_nonzero), dim=0)
        return pos

    dx_vals = _symmetric_range(max_dx, dx_step)
    dy_vals = _symmetric_range(max_dy, dy_step)
    dz_vals = _symmetric_range(max_dz, dz_step)

    # 3. 生成网格
    DX, DY, DZ = torch.meshgrid(dx_vals, dy_vals, dz_vals, indexing='ij')

    # 4. 加上基准位置
    translations_ecef = torch.stack([
        DX.flatten() + base_ecef[0],
        DY.flatten() + base_ecef[1],
        DZ.flatten() + base_ecef[2]
    ], dim=1)

    return translations_ecef  # [M, 3]

def euler_angles_to_matrix_ECEF_batch_with_translations(
    euler_angles,
    translations_ecef,
    base_translation_wgs84,
    device='cuda'
):
    """
    生成批量变换矩阵，支持角度和位置的笛卡尔积

    Args:
        euler_angles: (N, 3) 欧拉角 [pitch, roll, yaw] in degrees
        translations_ecef: (M, 3) ECEF 坐标
        base_translation_wgs84: (3,) WGS84 坐标 [lon, lat, h]，用于计算 ENU -> ECEF 旋转
        device: 'cuda' or 'cpu'

    Returns:
        T: (N*M, 4, 4) 变换矩阵
    """
    euler_angles = torch.as_tensor(euler_angles, device=device, dtype=torch.float32)
    translations_ecef = torch.as_tensor(translations_ecef, device=device, dtype=torch.float32)
    base_translation_wgs84 = torch.as_tensor(base_translation_wgs84, device=device, dtype=torch.float32)

    N = euler_angles.shape[0]
    M = translations_ecef.shape[0]

    # 计算旋转矩阵 (使用基准位置的 ENU -> ECEF，位置扰动较小时 ENU 变化可忽略)
    R_batch = euler_to_rotm_batch(euler_angles, base_translation_wgs84, degrees=True, device=device)  # (N, 3, 3)

    # 生成笛卡尔积: N 个角度 × M 个位置 = N*M 个位姿
    R_expanded = R_batch.unsqueeze(1).expand(N, M, 3, 3).reshape(N * M, 3, 3)
    t_expanded = translations_ecef.unsqueeze(0).expand(N, M, 3).reshape(N * M, 3)

    # 构造变换矩阵
    T = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(N * M, 1, 1)
    T[:, :3, :3] = R_expanded
    T[:, :3, 3] = t_expanded

    return T  # (N*M, 4, 4)
def preprocess_param_cuda(camera, pose, device='cuda'):
    # pose: torch tensor (4,4) or (N,4,4)
    pose = pose.to(device)
    pose[..., 0:3, 1] *= -1
    pose[..., 0:3, 2] *= -1

    # camera.c 也是 torch tensor
    # h = camera.size[1]
    # camera.c[1] = h - camera.c[1]

    return camera, pose

def get_3D_samples_v3(mkpts_r, depth_mat, T_c2w, camera, euler_angles, translation,  query_euler_angles, query_translation, last_frame_info = {}, origin = None, device = 'cuda', num_init_pose = 32, T_query_in_ECEF_c2w = None, mul = None):
    '''
    核心功能：
    1. 生成 Query 的初始候选位姿 (generate_rotvecs...)
    2. 对传入的 3D 点做坐标系转换 (EPSG:4547 -> ECEF)
    3. 对 3D 点和位姿做去中心化处理 (-origin, -dd)
    '''
    # --- 1. 生成初始位姿候选 (Query Pose Candidates) ---
    # 1.1 角度撒种子
    query_euler_angles_batch = generate_rotvecs_cuda_sym3d(
        base_pitch=query_euler_angles[0],
        base_roll=query_euler_angles[1],
        base_yaw=query_euler_angles[2],
        max_pitch=0.0, pitch_step=2.0,
        max_yaw=10.0, yaw_step=2.0,
        max_roll=0.0, roll_step=2.0
    )

    # 1.2 位置撒种子 (在 ECEF 坐标系下)
    # query_translation 是 WGS84 [lon, lat, h]，在 ECEF 下加扰动
    translations_ecef_batch = generate_translations_ecef_sym3d(
        base_translation_wgs84=query_translation,
        max_dx=10.0, dx_step=2.0,  # ECEF X 方向 (米), 设为0表示不撒种子
        max_dy=10.0, dy_step=2.0,  # ECEF Y 方向 (米), 设为0表示不撒种子
        max_dz=0.0, dz_step=2.0,  # ECEF Z 方向 (米), 设为0表示不撒种子
        device=device
    )
    # 1.3 生成变换矩阵 (角度 × 位置 的笛卡尔积)
    query_T_c2w = euler_angles_to_matrix_ECEF_batch_with_translations(
        euler_angles=query_euler_angles_batch,
        translations_ecef=translations_ecef_batch,
        base_translation_wgs84=query_translation,  # 用于计算 ENU -> ECEF 旋转
        device=device
    )

    # 坐标轴调整 (Y轴取反, Z轴取反) - 根据你的相机坐标系定义
    query_T_c2w[:, :3, 1] = -query_T_c2w[:, :3, 1]
    query_T_c2w[:, :3, 2] = -query_T_c2w[:, :3, 2]

    min_offset_angle = 0.0
    max_offset_angle = 0.0
    min_offset_translation = 0.0
    max_offset_translation = 0.0

    # --- 2. 预处理 Render 相机参数 ---
    # 保存原始的ECEF平移向量（用于深度损失计算）
    render_T_original_ecef = T_c2w[:3, 3].copy() if isinstance(T_c2w, np.ndarray) else T_c2w[:3, 3].clone()

    render_camera, render_T = preprocess_param_cuda(copy.deepcopy(camera), copy.deepcopy(T_c2w))
    cx, cy = render_camera.c
    fx, fy = render_camera.f
    render_K = torch.tensor([[fx, 0, cx],[0, fy, cy], [0, 0, 1]], device=device)
    K_c2w = render_K.inverse()
    render_T = torch.as_tensor(render_T, device=device)

    # 如果 mkpts_r 存在才转 tensor，否则跳过
    if mkpts_r is not None:
        mkpts_r = torch.as_tensor(mkpts_r, device=device)

    # --- 3. 处理 3D 点 (坐标转换) ---
    # 逻辑：只要传入了 depth_mat 且是点列表形状，就认为是预计算好的 3D 点
    use_precomputed_points = True

    if use_precomputed_points:
        points3d = torch.as_tensor(depth_mat, device=device, dtype=torch.float32)
        if points3d.dim() == 3:
            points3d = points3d.view(-1, 3)

        # [关键修改 1] 计算 Mask
        valid = torch.isfinite(points3d).all(dim=1) & (points3d[:, 2] > 0)

        # [关键修改 2] 同步过滤 3D 点
        points3d = points3d[valid]

        # [关键修改 3] 同步过滤 2D 点 (mkpts_r) !!!
        # 只有这里也过滤了，才能保证 499 对 499
        if mkpts_r is not None:
             # 1. 直接把前面乱七八糟的维度压扁，只保留点数(N)和坐标(2)
             # [1, 1, 500, 2] -> [500, 2]
             mkpts_r = mkpts_r.view(-1, 2)

             # 2. 现在是一维对应一维，直接过滤
             mkpts_r = mkpts_r[valid]

        if points3d.numel() == 0:
            # 如果点为空，返回空tensor防止崩溃，或者抛出异常
            print("Warning: No valid 3D points in get_3D_samples_v3")
            Points_3D_ECEF = torch.empty((0, 3), device=device)
        else:
            # 坐标转换: EPSG:4547 (投影坐标) -> EPSG:4978 (ECEF)
            # pyproj 仍运行在 CPU 上，但这里尽量只做一次 tensor->numpy 和一次 numpy->tensor。
            transformer_4547_to_4978 = _get_cached_transformer("EPSG:4547", "EPSG:4978")
            points3d_np = points3d.detach().cpu().numpy()
            X, Y, Z = transformer_4547_to_4978.transform(
                points3d_np[:, 0], points3d_np[:, 1], points3d_np[:, 2]
            )
            points3d_ecef_np = np.column_stack((X, Y, Z))
            Points_3D_ECEF = torch.as_tensor(
                points3d_ecef_np, device=device, dtype=torch.float32
            )

    else:
        # 旧逻辑：深度图反投影 (如果 mkpts_r 为 None 这里会报错，但现在的流程不会进这里)
        if mkpts_r is None:
            raise ValueError("mkpts_r cannot be None when not using precomputed points")
        depth, valid = read_valid_depth(mkpts_r, depth=depth_mat, device=device)
        Points_3D_ECEF = get_Points3D_torch_cuda(
            depth,
            render_T[:3, :3],
            render_T[:3, 3],
            K_c2w,
            mkpts_r[valid])

    # --- 4. 尺度缩放 (Mul) ---
    if mul is not None:
        Points_3D_ECEF = Points_3D_ECEF * mul
        render_T[:3, 3] = render_T[:3, 3] * mul
        max_offset_translation = max_offset_translation * mul
        if origin is not None:
            origin = origin * mul
        query_T_c2w[:, :3, 3] = query_T_c2w[:, :3, 3] * mul

    # --- 5. 去中心化 (Origin) ---
    if origin is None:
        # 如果没有指定原点，就用第一个点作为原点
        if Points_3D_ECEF.shape[0] > 0:
            origin = Points_3D_ECEF[0]
        else:
            origin = torch.tensor([0., 0., 0.], device=device)

    Points_3D_ECEF_origin = Points_3D_ECEF - origin

    # 更新 Render 和 Query 的位姿 (减去 origin)
    render_T[:3, 3] -= origin
    # 构造 Render 的 w2c
    render_T_c2w = Pose.from_Rt(render_T[:3, :3], render_T[:3, 3])
    T_render = render_T_c2w.inv().float()

    query_T_c2w[:, :3, 3] -= origin
    T_query = Pose.from_Rt(query_T_c2w[:, :3, :3], query_T_c2w[:, :3, 3]).inv() # w2c

    # --- 6. 归一化中心 (DD Center) ---
    center = True
    if center and Points_3D_ECEF_origin.shape[0] > 0:
        points3D_total = Points_3D_ECEF_origin
        points_max = points3D_total.max(dim=0)[0]
        points_min = points3D_total.min(dim=0)[0]
        points_size = points_max - points_min
        dd = points_min + points_size / 2
        Points_3D_ECEF_origin_center = points3D_total - dd

        # 调整位姿以适应 dd 偏移
        tt = T_render.t + T_render.R @ dd
        T_render = Pose.from_Rt(T_render.R, tt)

        tt = T_query.t + T_query.R @ dd
        T_query = Pose.from_Rt(T_query.R, tt)
    else:
        Points_3D_ECEF_origin_center = Points_3D_ECEF_origin
        dd = torch.tensor([0., 0., 0.], device=device)

    # --- 7. 生成初始候选位姿 (Initial Poses) ---
    if 'candidate_poses' not in last_frame_info.keys():
        T_query_c2w_obj = T_query.inv()
        # 注意：n 应该用 query_T_c2w 的数量（角度×位置的笛卡尔积）
        num_candidates = query_T_c2w.shape[0]
        random_aa, random_t = generate_random_aa_and_t_cuda(
            n = num_candidates,
            min_offset_angle = min_offset_angle,
            max_offset_angle = max_offset_angle,
            min_offset_translation = min_offset_translation,
            max_offset_translation = max_offset_translation
        )
        random_pose = Pose.from_aa(random_aa, random_t)
        initial_poses = T_query_c2w_obj @ random_pose
        initial_poses_w2c = initial_poses.inv()
    # else:
    #     # 如果有上一帧信息，可能复用 (这里保留你原本的逻辑接口)
        # initial_poses_w2c = last_frame_info['candidate_poses']

    # 将原始ECEF平移向量也返回（用于深度损失计算）
    render_T_original_ecef_tensor = torch.tensor(render_T_original_ecef, device=device, dtype=torch.float32)
    return Points_3D_ECEF_origin_center, T_render, initial_poses_w2c, dd, mkpts_r, render_T_original_ecef_tensor
def generate_render_camera(camera):
    if len(camera) == 5:
        image_width_px, image_height_px, sensor_width_mm, sensor_height_mm, f_mm = camera
        # 计算内参矩阵中的焦距
        focal_ratio_x = f_mm / sensor_width_mm
        focal_ratio_y = f_mm / sensor_height_mm

        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y

        # 计算主点坐标
        cx = image_width_px / 2
        cy = image_height_px / 2
    elif len(camera) == 7:
        image_width_px, image_height_px, cx, cy, sensor_width_mm, sensor_height_mm, f_mm = camera
        # 计算内参矩阵中的焦距
        focal_ratio_x = f_mm / sensor_width_mm
        focal_ratio_y = f_mm / sensor_height_mm

        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y
    elif len(camera) == 8:
        image_width_px, image_height_px, cx, cy, sensor_width_mm, sensor_height_mm, fx_mm, fy_mm = camera
        # 计算内参矩阵中的焦距
        focal_ratio_x = fx_mm / sensor_width_mm
        focal_ratio_y = fy_mm / sensor_height_mm

        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y
    elif len(camera) == 6:
        image_width_px, image_height_px, cx, cy, fx, fy= camera
        # 计算内参矩阵中的焦距
        focal_ratio_x = fx / image_width_px
        focal_ratio_y = fy / image_height_px

        fx = image_width_px * focal_ratio_x
        fy = image_height_px * focal_ratio_y
    camera_param = {
    'model': 'PINHOLE',
    'width': image_width_px,
    'height': image_height_px,
    'params': np.array([fx, fy, cx, cy])}
    cams = Camera.from_colmap(camera_param)

    return cams
def get_Points3D_torch_cuda(depth, R, t, K, points):
    """
    根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及图像上的二维点坐标和深度信息，
    计算对应的三维世界坐标。

    参数:
    - depth: 深度值数组，尺寸为 [n,]，其中 n 是点的数量。
    - R: 旋转矩阵，尺寸为 [3, 3]，表示从世界坐标系到相机坐标系的旋转。
    - t: 平移向量，尺寸为 [3, 1]，表示从世界坐标系到相机坐标系的平移。
    - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
    - points: 二维图像坐标数组，尺寸为 [n, 2]，其中 n 是点的数量。

    返回:
    - Points_3D: 三维世界坐标数组，尺寸为 [n, 3]。
    """
    # 检查points是否为同质坐标，如果不是则扩展为同质坐标
    if points.shape[-1] != 3:
        points_2D = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        points_2D = points_2D.T
    else:
        points_2D = points.T

    # 扩展平移向量以匹配点的数量
    t = t.unsqueeze(1)  # 这相当于np.expand_dims(t, -1)
    t = t.repeat(1, points_2D.size(-1))  # 这相当于np.tile(t, points_2D.shape[-1])

    # 将所有输入转换为高精度浮点数类型
    points_2D = points_2D.float()
    K = K.float()
    R = R.float()
    depth = depth.float()
    t = t.float()

    # 修改内参矩阵的最后一项，以适应透视投影
    # K[-1, -1] = -1

    # 计算三维世界坐标
    Points_3D = R @ (K @ (depth * points_2D)) + t

    # 返回三维点坐标，形状为 [n, 3]
    return Points_3D.T

def zero_pad(size, image):

    h, w = image.shape[:2]
    # 保证目标尺寸不小于原尺寸，避免 broadcast 报错
    target = max(size, h, w)
    padded = np.zeros((target, target)+image.shape[2:], dtype=image.dtype)
    padded[:h, :w] = image

    return padded
