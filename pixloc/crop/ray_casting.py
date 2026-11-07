import numpy as np
import os
from osgeo import gdal
import pyproj
from typing import Dict
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation as R

class TargetLocation():
    def __init__(self, config: Dict, use_dsm = False):
        self.config = config
        self.clicked_points = []
        self._k_inv_cache = {}
        self._alpha_cache = {}
        self._wgs84_to_proj = pyproj.Transformer.from_crs(
            "EPSG:4326", "EPSG:4547", always_xy=True
        )
        self._coord_transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        if use_dsm:
            DSM_path = self.config["ray_casting"]["DSM_path"]
            DSM_npy_path = self.config["ray_casting"]["DSM_npy_path"]
            geotransform_path = self.config["ray_casting"]["geotransform_path"]

            self.num_sample = self.config["ray_casting"]["num_sample"]
            self.area_minZ = self.config["ray_casting"]["area_minZ"]

            cur_dir = os.path.abspath(os.path.dirname(__file__))
            dsm_path = cur_dir + '/../' + DSM_path
            npy_path = cur_dir + '/../' + DSM_npy_path
            geotransform_path = cur_dir + "/../" + geotransform_path

            if os.path.isfile(npy_path):
                self.area = np.load(npy_path)
                self.geotransform = np.load(geotransform_path)
            else:
                self.area, self.geo_transform, self.area_minZ = self.dsm2npy(dsm_path, npy_path, geotransform_path)

    def _get_k_inv(self, K):
        K_arr = np.asarray(K, dtype=np.float64)
        key = K_arr.tobytes()
        if key not in self._k_inv_cache:
            self._k_inv_cache[key] = np.linalg.inv(K_arr)
        return self._k_inv_cache[key]

    def _get_alpha(self, num_sample):
        if num_sample not in self._alpha_cache:
            self._alpha_cache[num_sample] = np.linspace(
                0.0, 1.0, num_sample, dtype=np.float64
            )[None, :]
        return self._alpha_cache[num_sample]

    @staticmethod
    def dsm2npy(dsm_path, npy_path, geotransform_path):
        DSM_map = gdal.Open(dsm_path)
        band = DSM_map.GetRasterBand(1)
        geotransform = DSM_map.GetGeoTransform()
        area = band.ReadAsArray()
        del DSM_map
        mask = np.ma.masked_values(area, -9999)
        area_minZ = mask.min()
        np.save(npy_path, area)
        np.save(geotransform_path, geotransform)
        return area, geotransform, area_minZ

    def _build_camera_to_world(self, pose):
        raw_lon, raw_lat, raw_alt = pose[0], pose[1], pose[2]
        raw_roll, raw_pitch, raw_yaw = pose[3], pose[4], pose[5]

        x_meter, y_meter = self._wgs84_to_proj.transform(raw_lon, raw_lat)
        matrix_3x3 = R.from_euler(
            'xyz', [raw_pitch, raw_roll, raw_yaw], degrees=True
        ).as_matrix()

        T4x4 = np.eye(4, dtype=np.float64)
        T4x4[:3, :3] = matrix_3x3
        T4x4[:3, 3] = [x_meter, y_meter, raw_alt]
        return T4x4 @ self._coord_transform

    def caculate_predictXYZ_batch(self, K, pose, obj_pixel_coords, area, geotransform, area_minZ, num_sample):
        coords = np.asarray(obj_pixel_coords, dtype=np.float64).reshape(-1, 2)
        T = self._build_camera_to_world(pose)
        R_world = T[:3, :3]
        cam_center = T[:3, 3].astype(np.float64)

        K_inv = self._get_k_inv(K)
        pixels = np.concatenate(
            [coords, np.ones((coords.shape[0], 1), dtype=np.float64)], axis=1
        )
        p_camera = (K_inv @ pixels.T).T
        targets = (R_world @ p_camera.T).T + cam_center[None, :]

        ray_dirs = targets - cam_center[None, :]
        z_denom = ray_dirs[:, 2]
        z_denom = np.where(np.abs(z_denom) < 1e-12, 1e-12, z_denom)
        alpha = self._get_alpha(num_sample)
        scale_to_ground = (area_minZ - cam_center[2]) / z_denom
        intersections = cam_center[None, :] + ray_dirs * scale_to_ground[:, None]

        x = cam_center[0] + (intersections[:, 0:1] - cam_center[0]) * alpha
        y = cam_center[1] + (intersections[:, 1:2] - cam_center[1]) * alpha
        z_values = cam_center[2] + (intersections[:, 2:3] - cam_center[2]) * alpha

        x_origin, x_pixel_size, _, y_origin, _, y_pixel_size = geotransform
        col = ((x - x_origin) / x_pixel_size).astype(int)
        row = ((y - y_origin) / y_pixel_size).astype(int)
        sample_height = map_coordinates(area, [row, col], order=1)

        abs_diff = np.abs(z_values - sample_height)
        min_idx = np.argmin(abs_diff, axis=1)
        batch_idx = np.arange(coords.shape[0])

        result_x = x[batch_idx, min_idx]
        result_y = y[batch_idx, min_idx]
        result_z = z_values[batch_idx, min_idx]
        result_sample_height = sample_height[batch_idx, min_idx]

        origin_x = cam_center[0]
        target_x = targets[:, 0]
        target_y = targets[:, 1]
        origin_y = cam_center[1]
        use_x = np.abs(target_x - origin_x) > 1e-6
        k_value = np.empty(coords.shape[0], dtype=np.float64)
        k_value[use_x] = (result_x[use_x] - origin_x) / (target_x[use_x] - origin_x)
        k_value[~use_x] = (result_y[~use_x] - origin_y) / (target_y[~use_x] - origin_y)

        points = np.stack([result_x, result_y, result_z], axis=1)
        return points, k_value, result_sample_height

    def predict_points_alt(self, DSM_path, pose, ref_npy_path, geotransform, K,
                           ray_area, ray_area_minZ, num_sample, object_pixel_coords_list):
        area = ray_area
        area_minZ = ray_area_minZ
        points, _, _ = self.caculate_predictXYZ_batch(
            K, pose, object_pixel_coords_list, area, geotransform, area_minZ,
            num_sample=num_sample
        )
        return points
