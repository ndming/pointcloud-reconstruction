import cv2 as cv
import numpy as np
from os.path import basename


_ARUCO_DICTS = {
	"DICT_4X4_50":   cv.aruco.DICT_4X4_50,
	"DICT_4X4_100":  cv.aruco.DICT_4X4_100,
	"DICT_4X4_250":  cv.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
	"DICT_5X5_50":   cv.aruco.DICT_5X5_50,
	"DICT_5X5_100":  cv.aruco.DICT_5X5_100,
	"DICT_5X5_250":  cv.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
	"DICT_6X6_50":   cv.aruco.DICT_6X6_50,
	"DICT_6X6_100":  cv.aruco.DICT_6X6_100,
	"DICT_6X6_250":  cv.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
	"DICT_7X7_50":   cv.aruco.DICT_7X7_50,
	"DICT_7X7_100":  cv.aruco.DICT_7X7_100,
	"DICT_7X7_250":  cv.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5":  cv.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9":  cv.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}


def detect_dictionary(img):
    for dict_name, aruco_dict in _ARUCO_DICTS.items():
        dictionary = cv.aruco.getPredefinedDictionary(aruco_dict)
        params = cv.aruco.DetectorParameters()
        detector = cv.aruco.ArucoDetector(dictionary, params)
        corners, _, _ = detector.detectMarkers(img)

        if len(corners) > 0:
            print(f"Detected {len(corners)} markers for {dict_name}")


def match_image_points(
    image_files: list[str], 
    detector: cv.aruco.CharucoDetector,
    target_size: tuple[int, int]
):
    all_obj_points, all_img_points = [], []
    for file in image_files:
        image = cv.imread(file)
        image = cv.resize(image, target_size)

        # Detect markers and interpolate corners. If calibration parameters are 
        # not provided, the ChArUco corners are interpolated by calculating the 
        # corresponding homography between the ChArUco plane and image 
        # projection. After the ChArUco corners have been interpolated, a 
        # subpixel refinement is automatically performed.
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)

        if charuco_ids is None or len(charuco_ids) < 3:
            continue
        
        file_name = basename(file)
        print(f"[>] Detected {len(charuco_ids)} ChArUco corners in {file_name}")
        obj_points, img_points = detector.getBoard()\
            .matchImagePoints(charuco_corners, charuco_ids)

        if len(obj_points) < 4:
            print(f"[!] Point matching discarded for {file_name}")
            continue

        all_obj_points.append(obj_points)
        all_img_points.append(img_points)

    return all_obj_points, all_img_points


def estimate_extrinsic(
    image,
    camera_matrix, 
    distortion_coefficients,
    board: cv.aruco.CharucoBoard,
    img_size: tuple[int, int]
):
    charuco_params = cv.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = False
    detector_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.CharucoDetector(board, charuco_params, detector_params)

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(image)
    print(f"[>] Detected {len(charuco_ids)} ChArUco corners")
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)

    ret, _, _, rvecs, tvecs = cv.calibrateCamera(
        [obj_points], [img_points], img_size, camera_matrix, distortion_coefficients)
    
    print("---- Result ----")
    print(f"Rotation vectors:")
    print(rvecs[0])
    print(f"Translation vectors:")
    print(tvecs[0])
    print(f"Reprojection error:")
    print(ret)

    return (ret, rvecs[0], tvecs[0])