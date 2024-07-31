import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
from typing import Tuple
import open3d as o3d



def pfm_imread(filename: str) -> Tuple[np.ndarray, float]:
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale



def write_pfm(file:str, image:np.ndarray, scale:float=1) -> None:
    """
    Write a single-channel numpy array to a PFM file.
    
    Args:
        file (str): File path to save the PFM file to.
        image (numpy.ndarray): Single-channel image data to save. Expected in (H, W) format.
        scale (float): Scale to write in the PFM header; also indicates endianness.
    """
    if image.dtype != np.float32:
        raise TypeError('Image dtype must be float32.')

    # Determine system byte order
    endian = sys.byteorder

    if endian == 'little':
        scale = -scale  # PFM format uses negative scale for little-endian

    with open(file, 'wb') as f:
        # Write the PFM header for a single-channel image
        f.write(b'Pf\n')
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode('ascii'))
        f.write(f"{scale}\n".encode('ascii'))
        
        # The PFM format specifies that the image data should be bottom-to-top,
        # so we flip the image vertically before saving.
        np.flipud(image).tofile(f)



def resize_and_crop_image(input_folder:str, output_folder:str, target_width:int, target_height:int, crop_side:str='center') -> None:
    """
    Resize and crop images in a folder to a target resolution, cropping from a specified side.
    
    Parameters:
    - input_folder: Path to the folder containing the original images.
    - output_folder: Path where the resized and cropped images will be saved.
    - target_width: The target width for the cropped images.
    - target_height: The target height for the cropped images.
    - crop_side: Side from which the image should be cropped ('left', 'right', 'center').
    
    Returns:
    None
    """
    os.makedirs(output_folder, exist_ok=True)
    
    original_images_list = [f for f in os.listdir(input_folder)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for image_path in original_images_list:
        full_path = os.path.join(input_folder, image_path)
        image = cv2.imread(full_path)
        #rotate image 180
        image = cv2.rotate(image, cv2.ROTATE_180)
        
        if image is None:
            continue

        original_height, original_width = image.shape[:2]
        target_aspect = target_width / target_height
        original_aspect = original_width / original_height

        if original_aspect > target_aspect:
            new_width_original = int(target_aspect * original_height)
            if crop_side == 'left':
                x_offset = 0
            elif crop_side == 'right':
                x_offset = original_width - new_width_original
            else:  # 'center'
                x_offset = (original_width - new_width_original) // 2
            cropped_image = image[:, x_offset:x_offset + new_width_original]
        else:
            new_height_original = int(original_width / target_aspect)
            y_offset = (original_height - new_height_original) // 2
            cropped_image = image[y_offset:y_offset + new_height_original, :]

        resized_image = cv2.resize(cropped_image, (target_width, target_height))
        
        image_path = image_path.replace('.jpg', '.png')
        cv2.imwrite(os.path.join(output_folder, image_path), resized_image)



def disparity_to_depth(focal_length:int, baseline:int, disparity:np.ndarray) -> np.ndarray:
    """ to convert disparity to depth, we use the formula: depth = f * b / disparity
    """
    return focal_length * baseline / (disparity + 0.0001)



def depth_to_disparity(focal_length:int, baseline:int, depth:np.ndarray) -> np.ndarray:
    """
    Convert depth to disparity using the formula: disparity = (f * b) / depth
    Where:
      - f is the focal length in pixels
      - b is the baseline in the same units as the depth (mm in this case)
      - depth is in mm
    """
    # Avoid division by zero by adding a small constant to depth
    return focal_length * baseline / (depth + 0.0001)



def reduce_depth_map_resolution(depth_map:np.ndarray, scale_percent:int=50) -> np.ndarray:
    """
    Reduce the resolution of a depth map by a given percentage.
    """
    width = int(depth_map.shape[1] * scale_percent / 100)
    height = int(depth_map.shape[0] * scale_percent / 100)
    new_dim = (width, height)
    
    reduced_depth_map = cv2.resize(depth_map, new_dim, interpolation=cv2.INTER_AREA)
    
    return reduced_depth_map



def find_and_compute_transform(image_path1: str, image_path2: str, chessboard_size: tuple, imshow: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Finds chessboard corners and computes the transformation matrix between two images based on detected chessboard corners.

    Args:
    - image_path1 (str): Path to the first image.
    - image_path2 (str): Path to the second image.
    - chessboard_size (Tuple[int, int]): Dimensions of the chessboard (number of internal corners by width and height).
    - imshow (bool): If True, display the images with detected corners.

    Returns:
    - Optional[Tuple[np.ndarray, np.ndarray]]: A tuple containing the relative rotation vector and translation vector if corners are found in both images; otherwise, None.
    """
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    ret1: bool
    corners1: np.ndarray
    ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)

    ret2: bool
    corners2: np.ndarray
    ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

    if not ret1 or not ret2:
        print("Chessboard corners not found in one or both images.")
        return None

    # Refine the corner positions
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
    corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

    # Assume the object points, same as your setup, based on the chessboard dimensions
    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    _: bool
    rvec1: np.ndarray # formen af rvec1: (3, 1)
    tvec1: np.ndarray # formen af rvec1: (3, 1)
    _, rvec1, tvec1 = cv2.solvePnP(objp, corners1, np.eye(3, 3), None)

    _: bool
    rvec2: np.ndarray # formen af rvec1: (3, 1)
    tvec2: np.ndarray # formen af rvec1: (3, 1)
    _, rvec2, tvec2 = cv2.solvePnP(objp, corners2, np.eye(3, 3), None)

    # Calculate relative rotation and translation vectors
    R1: np.ndarray    # formen af R1: (3, 3)
    R2: np.ndarray    # formen af R1: (3, 3)
    R_rel: np.ndarray # formen af R_rel: (3, 3)
    t_rel: np.ndarray # formen af t_rel: (3, 1)

    # Lineær algebra time:
    # Så har vi R1, R2, tvec1 og tvec2
    # Vi vil gerne finde R_rel som er rotationen fra kamera 1 til kamera 2
    # Og t_rel som er translationen eller position skiftet fra kamera 1 til kamera 2

    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_rel = R2 @ R1.T
    t_rel = tvec2 - R_rel @ tvec1

    rvec_rel: np.ndarray # shape of rvec_rel: (3, 1)
    rvec_rel, _ = cv2.Rodrigues(R_rel)

    # false i jupyter please
    if imshow:
        cv2.drawChessboardCorners(image1, chessboard_size, corners1, ret1)
        cv2.drawChessboardCorners(image2, chessboard_size, corners2, ret2)
        cv2.imshow('Chessboard 1', image1)
        cv2.imshow('Chessboard 2', image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return rvec_rel, t_rel



def find_homography_chessboard(image1:np.ndarray, image2:np.ndarray, chessboard_size:Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret1, corners1 = cv2.findChessboardCorners(gray1, chessboard_size, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, chessboard_size, None)

    if not ret1 or not ret2:
        print("Chessboard corners not found in one or both images.")
        return None

    # Refine the corner positions
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
    corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

    # Find homography
    H, mask = cv2.findHomography(corners1, corners2, cv2.RANSAC)

    return H, mask



def warp_image(image:np.ndarray, H:np.ndarray, dimensions:Tuple[int, int]) -> np.ndarray:
    # dimensions should be the size of the reference image
    warped_image = cv2.warpPerspective(image, H, dimensions)
    return warped_image



def generate_original_image_mask(image:np.ndarray, H:np.ndarray, dimensions:Tuple[int, int]) -> np.ndarray:
    """
    Generate a mask for the original image showing which pixels remain inside the frame
    of the warped image after applying the homography H.
    
    Args:
    - image (np.array): Original image to be transformed.
    - H (np.array): Homography matrix used for warping.
    - dimensions (tuple): Width and height of the warped image frame.

    Returns:
    - np.array: Mask where 1 indicates the pixel is visible in the warped image, 0 otherwise.
    """
    height, width = image.shape[:2]
    # Create grid of coordinates in the original image
    y_indices, x_indices = np.indices((height, width))
    ones = np.ones((height, width))
    indices = np.stack((x_indices, y_indices, ones), axis=-1).reshape((height * width, 3))
    
    # Apply homography to the coordinates
    transformed_indices = (H @ indices.T).T
    # Normalize coordinates
    transformed_indices[:, 0] /= transformed_indices[:, 2]
    transformed_indices[:, 1] /= transformed_indices[:, 2]
    
    # Check if the coordinates are within the valid range of the warped image dimensions
    valid_x = (transformed_indices[:, 0] >= 0) & (transformed_indices[:, 0] < dimensions[0])
    valid_y = (transformed_indices[:, 1] >= 0) & (transformed_indices[:, 1] < dimensions[1])
    valid_indices = valid_x & valid_y

    # Reshape back to image shape
    mask = valid_indices.reshape((height, width)).astype(np.uint8)
    
    return mask



def apply_transformation_to_point_cloud(pcd, rvec, tvec):
    """ Apply a 3D rotation and translation to a point cloud. """
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)  # Create a 4x4 transformation matrix
    T[:3, :3] = R  # Set the upper 3x3 part as the rotation matrix
    T[:3, 3] = tvec.squeeze()  # Set the translation

    # Apply the transformation
    pcd.transform(T)  # Open3D's point cloud transformation method
    return pcd



def apply_advanced_perspective_correction(pcd, homography_matrix):
    """ Apply a full perspective transformation to a 3D point cloud based on a 2D homography matrix. """
    points = np.asarray(pcd.points)
    # Assuming the z-coordinate is zero for simplification, you may need to adjust this assumption
    points_homogeneous = np.hstack((points[:, :2], np.ones((points.shape[0], 1))))
    transformed_points = (homography_matrix @ points_homogeneous.T).T
    transformed_points /= transformed_points[:, 2:3]  # Normalize by the homogeneous coordinate
    pcd.points = o3d.utility.Vector3dVector(np.hstack((transformed_points[:, :2], points[:, 2:3])))  # Keep original Z
    return pcd



def reduce_depth_map_resolution(depth_map, scale_percent=50):
    width = int(depth_map.shape[1] * scale_percent / 100)
    height = int(depth_map.shape[0] * scale_percent / 100)
    new_dim = (width, height)
    
    reduced_depth_map = cv2.resize(depth_map, new_dim, interpolation=cv2.INTER_AREA)
    
    return reduced_depth_map