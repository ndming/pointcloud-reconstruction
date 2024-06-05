import cv2 as cv
import numpy as np
import os
import glob
from datetime import datetime
import open3d as o3d
import matplotlib.pyplot as plt
import argparse

VISUAL_SIZE = (696, 522)

def show_image_pairs(images_1, images_2):
    for image_1, image_2 in zip(images_1, images_2):
        img_1 = cv.resize(image_1, VISUAL_SIZE)
        img_2 = cv.resize(image_2, VISUAL_SIZE)
        
        if img_1.dtype != img_2.dtype:
            img_2 = img_2.astype(img_1.dtype)

        image = cv.hconcat([img_1, img_2])

        cv.imshow("image_pair", image)

        key = cv.waitKey(0) & 0xFF

        if key == 13:  # ENTER key
            break
        if key == 32:  # SPACE key
            continue
        if key == 27:  # ESC key
            cv.destroyAllWindows()
            raise InterruptedError("ESC pressed")
    
    cv.destroyAllWindows()

def calibrate_camera(images_folder, aruco_dict_type=cv.aruco.DICT_6X6_1000, squares_x=5, squares_y=7, square_length=0.04, marker_length=0.02):
    images_names = sorted(glob.glob(images_folder + '*'))
    images = [cv.imread(imname) for imname in images_names]

    aruco_dict = cv.aruco.getPredefinedDictionary(aruco_dict_type)
    board = cv.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

    all_corners = []
    all_ids = []
    img_size = None

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        corners, ids, _ = cv.aruco.detectMarkers(gray, aruco_dict)
        if ids is not None:
            ret, charuco_corners, charuco_ids = cv.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ret > 0:
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

    if len(all_corners) > 0:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, img_size, None, None)

        return ret, camera_matrix, dist_coeffs

    print('No Charuco corners detected.')
    return None, None, None

def get_file_by_index(folder_path, index):
    files = sorted(glob.glob(folder_path))
    if index >= len(files):
        raise IndexError("Index out of range")
    return files[index]

def drawlines(img1, img2, lines, pts1, pts2):
    r, c, _ = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 5)
        img1 = cv.circle(img1, (int(pt1[0][0]), int(pt1[0][1])), 5, color, -1)
        img2 = cv.circle(img2, (int(pt2[0][0]), int(pt2[0][1])), 5, color, -1)
    return img1, img2

def draw_horizontal_lines(image, num_lines=20, color=(0, 255, 0), thickness=5):
    h, w = image.shape[:2]
    interval = h // num_lines
    for i in range(0, h, interval):
        image = cv.line(image, (0, i), (w, i), color, thickness)
    return image

def main(source_set):
    ret1, mtx1, dist1 = calibrate_camera(images_folder=f"{source_set}/calib/right/*")
    ret2, mtx2, dist2 = calibrate_camera(images_folder=f"{source_set}/calib/front/*")

    imgL = cv.imread(f"{source_set}/front.JPG")
    imgR = cv.imread(f"{source_set}/right.JPG")

    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    image_size = grayL.shape[::-1]

    sift = cv.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(grayL, None)
    kp2, desc2 = sift.detectAndCompute(grayR, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    E, mask = cv.findEssentialMat(pts1, pts2, mtx1, method=cv.RANSAC, prob=0.99, threshold=1.0)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    _, R, t, mask = cv.recoverPose(E, pts1, pts2, mtx1)

    R1, R2, P1, P2, _, _, _ = cv.stereoRectify(mtx1, None, mtx2, None, image_size, R, t)

    mapx1, mapy1 = cv.initUndistortRectifyMap(mtx1, None, R1, P1, image_size, cv.CV_32FC1)
    mapx2, mapy2 = cv.initUndistortRectifyMap(mtx2, None, R2, P2, image_size, cv.CV_32FC1)

    rectified_imgL = cv.remap(imgL, mapx1, mapy1, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    rectified_imgR = cv.remap(imgR, mapx2, mapy2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    K_inv = np.linalg.inv(mtx1)
    F = K_inv.T @ E @ K_inv
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)

    imgL_lines1, imgR_lines1 = drawlines(imgL.copy(), imgR.copy(), lines1, pts1, pts2)
    imgR_lines2, imgL_lines2 = drawlines(imgR.copy(), imgL.copy(), lines2, pts2, pts1)

    rectified_imgL_with_lines = draw_horizontal_lines(rectified_imgL.copy())
    rectified_imgR_with_lines = draw_horizontal_lines(rectified_imgR.copy())

    # Stereo matching parameters
    min_disparity = 0
    num_disparity = 16 * 5
    SADWindowSize = 5
    uniqueness = 5
    speckle_windows_size = 200
    speckle_range = 1
    max_disparity = min_disparity + num_disparity

    # Stereo matching
    left_matcher = cv.StereoSGBM_create(minDisparity=min_disparity, 
                                        numDisparities=num_disparity, 
                                        blockSize=SADWindowSize,
                                        P1=8 * 3 * SADWindowSize ** 2, 
                                        P2=32 * 3 * SADWindowSize ** 2, 
                                        uniquenessRatio=uniqueness,
                                        disp12MaxDiff=2, 
                                        speckleWindowSize=speckle_windows_size, 
                                        speckleRange=speckle_range)

    left_disparity = left_matcher.compute(rectified_imgL, rectified_imgR)
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    right_disparity = right_matcher.compute(rectified_imgR, rectified_imgL)

    # WLS filtering
    sigma = 1.5
    lambda_ = 8000
    wls = cv.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(lambda_)
    wls.setSigmaColor(sigma)
    filtered_disparity = wls.filter(left_disparity, rectified_imgL, disparity_map_right=right_disparity)
    cv.filterSpeckles(filtered_disparity, 0, 400, max_disparity - 5)
    _, filtered_disparity = cv.threshold(filtered_disparity, 0, max_disparity * 16, cv.THRESH_TOZERO)
    filtered_disparity = (filtered_disparity / 16).astype(np.uint8)

    b = np.linalg.norm(t)

    depth_map = mtx1[0, 0] * b / (filtered_disparity)
    depth_map = depth_map.astype('uint16')
    
    # Reprojection matrix
    Q = np.float32([[1, 0, 0, -mtx1[0, 2]],
                    [0, 1, 0, -mtx1[1, 2]],
                    [0, 0, 0, mtx1[0, 0]],
                    [0, 0, -1 / b, (mtx1[0, 2] - mtx2[0, 2]) / b]])

    points = cv.reprojectImageTo3D(filtered_disparity, Q)
    points = points.reshape(-1, 3)
    color = rectified_imgL.reshape(-1, 3)
    color = np.flip(color, axis=1) / 255
    xyzrgb = np.concatenate((points, color), axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])

    # Get the total number of points in the point cloud
    num_points = len(pcd.points)

    # Check if the number of points is greater than 2 million
    if num_points > 2_000_000:
        # Randomly sample 2 million points
        indices = np.random.choice(num_points, 2_000_000, replace=False)
        downsampled_pcd = pcd.select_by_index(indices)
    else:
        downsampled_pcd = pcd

    # Save the downsampled point cloud
    o3d.io.write_point_cloud(f'cloud.ply', downsampled_pcd)

    show_image_pairs([imgL_lines1, rectified_imgL_with_lines], [imgR_lines2, rectified_imgR_with_lines])
    show_image_pairs([filtered_disparity], [depth_map])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration and point cloud processing script")
    parser.add_argument('--source-set', type=str, required=True, help='Path to the folder containing calibration images')
    args = parser.parse_args()

    main(args.source_set)
