import cv2
# TODO: consider using PIL instead of OpenCV as it is heavy and only used here
import torch
from tqdm import tqdm
from ..utils import  undistort

    # Show the visualization
    # plt.show()
def numpy_image_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy(image / 255.).float()

def read_image(path, grayscale=False, scale = None, distortion =None, query_camera=None):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if distortion:
        image= undistort.main(image, query_camera, distortion)
    if image is None:
        raise IOError(f'Could not read image at {path}.')
    if not grayscale:
        image = image[..., ::-1]
    if scale:
        height, width = image.shape[:2]
        new_dim = (int(width // scale), int(height // scale))
        image = cv2.resize(image, new_dim)

    return image
def read_image_list(path_list, grayscale=False, scale = None, distortion = None, query_camera = None):
    image_list = []
    for path in tqdm(path_list):
        image = read_image(path, scale = scale, distortion=distortion, query_camera=query_camera)
        image_list.append(image)
    return image_list
def resize(image, size, fn=None, interp='linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h*scale)), int(round(w*scale))
        # TODO: we should probably recompute the scale like in the second case
        scale = (scale, scale)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f'Incorrect new size: {size}')
    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST}[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale
