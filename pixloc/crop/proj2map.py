import numpy as np
import time

import os
from PIL import Image
import cv2

def geo_coords_to_dsm_index(x, y, transform):
    col, row = ~transform * (x, y)
    return int(row), int(col)

def crop_dsm_dom_point(DSM_path, pose, ref_npy_path, geotransform, dom_data, dsm_data, dsm_transform,
                 K, pose_w2c, image_points, area_minZ, ray_area, ray_area_minZ,
                 crop_padding=10, locator=None):
    t0 = time.perf_counter()
    if locator is None:
        from .ray_casting import TargetLocation
        locator = TargetLocation({"ray_casting": {}}, use_dsm=False)

    world_points = locator.predict_points_alt(
        DSM_path, pose, ref_npy_path, geotransform, K,
        ray_area, ray_area_minZ, num_sample=4000,
        object_pixel_coords_list=image_points
    )
    t_raycast = time.perf_counter()
    dsm_indices = [
        geo_coords_to_dsm_index(float(xyz[0]), float(xyz[1]), dsm_transform)
        for xyz in world_points
    ]

    rows, cols = zip(*dsm_indices)
    dsm_height, dsm_width = dsm_data.shape

    row_min = max(min(rows) - crop_padding, 0)
    row_max = min(max(rows) + crop_padding, dsm_height)
    col_min = max(min(cols) - crop_padding, 0)
    col_max = min(max(cols) + crop_padding, dsm_width)

    dsm_crop = dsm_data[row_min:row_max, col_min:col_max]
    dom_crop = dom_data[:, row_min:row_max, col_min:col_max]

    xs = np.arange(col_min, col_max) * dsm_transform.a + dsm_transform.c
    ys = np.arange(row_min, row_max) * dsm_transform.e + dsm_transform.f
    xx, yy = np.meshgrid(xs, ys)

    dsm_poly_local = []
    for r, c in dsm_indices:
        dsm_poly_local.append([c - col_min, r - row_min])
    dsm_poly_local = np.array(dsm_poly_local, dtype=np.float32)

    width_top = np.linalg.norm(dsm_poly_local[1] - dsm_poly_local[0])
    width_bottom = np.linalg.norm(dsm_poly_local[2] - dsm_poly_local[3])
    height_left = np.linalg.norm(dsm_poly_local[3] - dsm_poly_local[0])
    height_right = np.linalg.norm(dsm_poly_local[2] - dsm_poly_local[1])
    out_w = max(int(round(max(width_top, width_bottom))), 1)
    out_h = max(int(round(max(height_left, height_right))), 1)

    dst_corners = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(dsm_poly_local, dst_corners)

    dom_hwc = np.transpose(dom_crop, (1, 2, 0))
    dom_warp = cv2.warpPerspective(dom_hwc, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    dom_warp = np.transpose(dom_warp, (2, 0, 1))

    dsm_warp = cv2.warpPerspective(dsm_crop.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)

    x_warp = cv2.warpPerspective(xx.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    y_warp = cv2.warpPerspective(yy.astype(np.float32), H, (out_w, out_h), flags=cv2.INTER_LINEAR)

    xyz = np.stack([x_warp, y_warp, dsm_warp], axis=-1)
    xyz = np.nan_to_num(xyz, nan=0.0)
    t_warp = time.perf_counter()

    dsm_indices = [(col, row) for row, col in dsm_indices]
    timings = {
        "raycast_s": t_raycast - t0,
        "warp_s": t_warp - t_raycast,
        "total_s": t_warp - t0,
    }
    return dom_warp, xyz, np.array(dsm_indices), world_points.tolist(), min(out_w, out_h), timings

def generate_ref_map(DSM_path, pose, ref_npy_path, geotransform, query_intrinsics, query_poses, name,
                     area_minZ, dsm_data, dsm_transform, dom_data, ref_rgb_path, ref_depth_path,
                     ray_area, ray_area_minZ, crop_padding=2, debug=True, save_to_disk=True,
                     locator=None):

    K_w2c = query_intrinsics
    pose_w2c = query_poses
    output_img_path = os.path.join(ref_rgb_path, f'{name}_dom.png')
    output_npy_path = os.path.join(ref_depth_path, f'{name}_dsm.npy')
    width, height = K_w2c[0,2]*2, K_w2c[1,2]*2
    image_points = [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)]

    dom_crop, point_cloud_crop, dsm_indices, world_coords, min_shape, timings = crop_dsm_dom_point(
        DSM_path, pose, ref_npy_path, geotransform,
        dom_data, dsm_data, dsm_transform,
        K_w2c, pose_w2c, image_points, area_minZ, ray_area, ray_area_minZ, crop_padding,
        locator=locator
    )

    if debug:
        import cv2

        dom_vis = np.transpose(dom_data, (1, 2, 0)).copy()
        cv2.polylines(dom_vis, [dsm_indices.reshape((-1, 1, 2))],
                    isClosed=True, color=(255, 0, 0), thickness=20)

        for idx, (x, y) in enumerate(dsm_indices):
            cv2.circle(dom_vis, (x, y), radius=20, color=(0, 255, 255), thickness=-1)
            cv2.putText(dom_vis, str(idx), (x + 5, y - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=10,
            color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        scale = 0.2
        new_size = (int(dom_vis.shape[1] * scale), int(dom_vis.shape[0] * scale))

        dom_vis_resized = cv2.resize(dom_vis, new_size, interpolation=cv2.INTER_AREA)

        os.makedirs(os.path.join("/home/amax/Documents/datasets/crop/DJI_20251217153719_0002_V",'debug_view'), exist_ok=True)
        save_path = os.path.join("/home/amax/Documents/datasets/crop/DJI_20251217153719_0002_V",'debug_view',name+'_view.png')

        cv2.imwrite(save_path, cv2.cvtColor(dom_vis_resized, cv2.COLOR_RGB2BGR))

    cropped_dom_img_uint8 = np.clip(dom_crop, 0, 255).astype(np.uint8)

    if cropped_dom_img_uint8.shape[0] == 3:
        cropped_dom_img_uint8 = np.transpose(cropped_dom_img_uint8, (1, 2, 0))

    if save_to_disk:
        Image.fromarray(cropped_dom_img_uint8).save(output_img_path)
        np.save(output_npy_path, point_cloud_crop)

    data = {
            'imgr_name':name+'_dom',
            'exr_name':name+'_dsm',
            'min_shape': min_shape,
            'dom_crop': cropped_dom_img_uint8,
            'point_cloud_crop': point_cloud_crop,
            'timings': timings,
        }
    return data
