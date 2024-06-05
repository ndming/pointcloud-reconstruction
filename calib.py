from cv2 import aruco
import numpy as np


def match_stereo_charuco(
    l_images: list[np.array], 
    r_images: list[np.array],
    detector: aruco.CharucoDetector
) -> tuple[list[np.array], list[np.array], list[np.array], 
           list[np.array], list[np.array], np.array]:
    obj_point_sets = []
    l_img_point_sets = []
    r_img_point_sets = []
    l_corner_sets = []
    r_corner_sets = []
    included_mask = np.ones(len(l_images), dtype=bool)

    for i, (l_image, r_image) in enumerate(zip(l_images, r_images)):
        l_corners, l_ids, _, _ = detector.detectBoard(l_image)
        r_corners, r_ids, _, _ = detector.detectBoard(r_image)

        if l_ids is None or r_ids is None:
            included_mask[i] = 0
            continue

        # Find common IDs
        common_ids = set(l_ids.flatten()).intersection(set(r_ids.flatten()))
        if not common_ids:
            included_mask[i] = 0
            continue

        # Filter out corners that are common in both images
        l_common = np.isin(l_ids.flatten(), list(common_ids), True)
        r_common = np.isin(r_ids.flatten(), list(common_ids), True)

        l_corners = l_corners[l_common]
        r_corners = r_corners[r_common]
        ids = l_ids[l_common]

        board = detector.getBoard()
        obj_points, l_img_points = board.matchImagePoints(l_corners, ids)
        _, r_img_points = board.matchImagePoints(r_corners, ids)

        obj_point_sets.append(obj_points)
        l_img_point_sets.append(l_img_points)
        r_img_point_sets.append(r_img_points)
        l_corner_sets.append(l_corners)
        r_corner_sets.append(r_corners)
        
    return (obj_point_sets, l_img_point_sets, r_img_point_sets, 
            l_corner_sets, r_corner_sets, included_mask.astype(bool))