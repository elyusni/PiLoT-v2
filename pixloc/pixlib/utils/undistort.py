import cv2
import numpy as np

def read_intrinsics(camera):
    """
    计算35mm等效焦距和内参矩阵。

    参数:
    image_width_px -- 图像的宽度（像素）
    image_height_px -- 图像的高度（像素）
    sensor_width_mm -- 相机传感器的宽度（毫米）
    sensor_height_mm -- 相机传感器的高度（毫米）

    返回:
    K -- 内参矩阵，形状为3x3
    """
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
    # 构建内参矩阵 K
    K = [[fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]]

    return np.array(K), image_width_px, image_height_px

def main(image,
         query_camera,
         kp,
         ):
    kp = np.array(kp).astype(np.float32)
    CameraMatrix, w, h = read_intrinsics(query_camera)
    img_disort = cv2.undistort(image, CameraMatrix, kp)
    cv2.imwrite('/home/ubuntu/Documents/code/FPVLoc/object_index/undistorted.jpg', img_disort)
    return img_disort
