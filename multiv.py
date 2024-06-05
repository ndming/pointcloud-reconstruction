from argparse import ArgumentParser
from os.path import abspath
from pathlib import Path
from tqdm import tqdm

import cv2 as cv
import numpy as np
import open3d as o3d

from calib import match_stereo_charuco
from dense import extract_window, match_template, TM_SQDIFF_NORMED
from dimen import BOARD_SQ_SIZE, SQUARE_LENGTH, MARKER_LENGTH, TARGET_SIZE, VISUAL_SIZE
from draws import draw_stereo_matching, draw_stereo_epilines, draw_lines
from utils import prompt_image_point, build_mask
from valid import is_convex, is_ccw


POINT_RADIUS = 12
LINE_THICKNESS = 5


def show_image_pairs(images_1, images_2):
    for image_1, image_2 in zip(images_1, images_2):
        img_1 = cv.resize(image_1, VISUAL_SIZE)
        img_2 = cv.resize(image_2, VISUAL_SIZE)
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


def show_image_quads(images_1, images_2, images_3, images_4):
    iterable = zip(images_1, images_2, images_3, images_4)
    for image_1, image_2, image_3, image_4 in iterable:
        img_1 = cv.resize(image_1, (np.array(VISUAL_SIZE) // 1.5).astype(int))
        img_2 = cv.resize(image_2, (np.array(VISUAL_SIZE) // 1.5).astype(int))
        img_3 = cv.resize(image_3, (np.array(VISUAL_SIZE) // 1.5).astype(int))
        img_4 = cv.resize(image_4, (np.array(VISUAL_SIZE) // 1.5).astype(int))
        pair_1 = cv.hconcat([img_1, img_2])
        pair_2 = cv.hconcat([img_3, img_4])
        quad = cv.vconcat([pair_1, pair_2])

        cv.imshow("image_quad", quad)

        key = cv.waitKey(0) & 0xFF
        if key == 13:  # ENTER key
            break
        if key == 32:  # SPACE key
            continue
        if key == 27:  # ESC key
            cv.destroyAllWindows()
            raise InterruptedError("ESC pressed")
    
    cv.destroyAllWindows()


def match_and_show_calib_patterns(calib_images_1, calib_images_2, detector, flipped=False):
    obj_point_sets, img_point_sets_1, img_point_sets_2, corner_sets_1, corner_sets_2, included = match_stereo_charuco(
        calib_images_1, calib_images_2, detector)
    calib_images_included_1 = np.array(calib_images_1)[included]
    calib_images_included_2 = np.array(calib_images_2)[included]

    if calib_images_included_1.size == 0 or calib_images_included_2.size == 0:
        raise RuntimeError("could not detect any common calib patterns")
    
    corner_sets_flattened_1 = [corners.reshape(-1, 2) for corners in corner_sets_1]
    corner_sets_flattened_2 = [corners.reshape(-1, 2) for corners in corner_sets_2]
    drawings_1, drawings_2 = draw_stereo_matching(
        calib_images_included_1, calib_images_included_2,
        corner_sets_flattened_1, corner_sets_flattened_2, radius=POINT_RADIUS)
    
    if not flipped:
        show_image_pairs(drawings_1, drawings_2)
    else:
        show_image_pairs(drawings_2, drawings_1)

    return (obj_point_sets, img_point_sets_1, img_point_sets_2, 
            corner_sets_1, corner_sets_2, 
            calib_images_included_1, calib_images_included_2)


def calibrate_and_show_stereo(obj_point_sets, img_point_sets_1, img_point_sets_2, 
    corner_sets_1, corner_sets_2, calib_images_included_1, calib_images_included_2, flipped=False):
    _, mtx_1, dist_1, _, _ = cv.calibrateCamera(obj_point_sets, img_point_sets_1, TARGET_SIZE, None, None)
    e, mtx_2, dist_2, _, _ = cv.calibrateCamera(obj_point_sets, img_point_sets_2, TARGET_SIZE, None, None)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = cv.CALIB_FIX_INTRINSIC
    error, _, _, _, _, R, t, E, F = cv.stereoCalibrate(
        obj_point_sets, img_point_sets_1, img_point_sets_2, mtx_1, dist_1, 
        mtx_2, dist_2, TARGET_SIZE, flags=flags, criteria=criteria)
    
    epiline_sets_1 = [cv.computeCorrespondEpilines(corners[0:4], whichImage=2, F=F) for corners in corner_sets_2]
    epiline_sets_1 = [lines.reshape(-1, 3) for lines in epiline_sets_1]
    corner_sets_flattened_2 = [corners[0:4].reshape(-1, 2) for corners in corner_sets_2]
    epipoint_matches_2, epiline_matches_1 = draw_stereo_epilines(
        calib_images_included_2, calib_images_included_1, 
        corner_sets_flattened_2, epiline_sets_1, radius=POINT_RADIUS, thickness=LINE_THICKNESS)
    
    epiline_sets_2 = [cv.computeCorrespondEpilines(corners[0:4], whichImage=1, F=F) for corners in corner_sets_1]
    epiline_sets_2 = [lines.reshape(-1, 3) for lines in epiline_sets_2]
    corner_sets_flattened_1 = [corners[0:4].reshape(-1, 2) for corners in corner_sets_1]
    epipoint_matches_1, epiline_matches_2 = draw_stereo_epilines(
        calib_images_included_1, calib_images_included_2, 
        corner_sets_flattened_1, epiline_sets_2, radius=POINT_RADIUS, thickness=LINE_THICKNESS)
    
    if not flipped:
        show_image_quads(epipoint_matches_1, epiline_matches_2, epiline_matches_1, epipoint_matches_2)
    else:
        show_image_quads(epipoint_matches_2, epiline_matches_1, epiline_matches_2, epipoint_matches_1)

    return mtx_1, mtx_2, error, R, t, E, F


def trifocal_callback(event, x, y, _, param):
    if event == cv.EVENT_LBUTTONDOWN:
        n_cols, _ = TARGET_SIZE
        v_cols, _ = VISUAL_SIZE
        
        f_u = x * (n_cols - 1.) / (v_cols - 1.)
        f_v = y * (n_cols - 1.) / (v_cols - 1.)

        f_win_name, f_image, r_win_name, r_image, l_win_name, l_image, fr_F, fl_F, rl_F, fl_P, lf_P, point, color = param
        f_clone = np.copy(f_image)
        r_clone = np.copy(r_image)
        l_clone = np.copy(l_image)

        cv.circle(f_clone, (int(f_u), int(f_v)), POINT_RADIUS, (0, 255, 0), -1)

        f_point = np.array([f_u, f_v])
        match = match_trifocal(f_point, f_image, r_image, l_image, fr_F, fl_F, rl_F)
        if match is None:
            return
        
        r_line, l_line, r_point_match, l_point_match, l_line_match = match
        draw_lines(r_clone, [r_line], (0, 255, 0), LINE_THICKNESS)
        draw_lines(l_clone, [l_line], (0, 255, 0), LINE_THICKNESS)

        cv.circle(r_clone, r_point_match.astype(int), POINT_RADIUS, (255, 0, 0), -1)
        draw_lines(l_clone, [l_line_match], (255, 0, 0), LINE_THICKNESS)
        cv.circle(l_clone, l_point_match.astype(int), POINT_RADIUS, (0, 0, 255), -1)

        cv.imshow(r_win_name, cv.resize(r_clone, VISUAL_SIZE))
        cv.imshow(l_win_name, cv.resize(l_clone, VISUAL_SIZE))
        cv.imshow(f_win_name, cv.resize(f_clone, VISUAL_SIZE))

        t_point = cv.triangulatePoints(
            fl_P, lf_P, f_point.reshape(2, 1), l_point_match.reshape(2, 1))
        t_point = t_point.flatten()
        if t_point[3] < 0:
            point[:] = np.nan
            color[:] = np.nan
            print("[!] unable to triangulate correspondence!")
        else:
            point[0:3] = t_point[0:3] / t_point[3]
            color[0:3] = np.flip(f_image[int(f_v), int(f_u), :])
            print(f"[>] triangulated point: {point}")


def match_trifocal(f_point, f_image, r_image, l_image, fr_F, fl_F, rl_F):
    n_cols, n_rows = TARGET_SIZE

    r_line = cv.computeCorrespondEpilines(f_point.reshape(1, 1, 2), whichImage=1, F=fr_F)
    r_line = r_line.reshape(-1, 3)[0]

    l_line = cv.computeCorrespondEpilines(f_point.reshape(1, 1, 2), whichImage=1, F=fl_F)
    l_line = l_line.reshape(-1, 3)[0]

    r_u = np.arange(0, n_cols, 2).astype(int)
    r_v = (-(r_line[2] + r_line[0] * r_u) / r_line[1]).astype(int)
    r_points = np.vstack([r_u, r_v]).T

    l_lines = cv.computeCorrespondEpilines(
        r_points.reshape(-1, 1, 2), whichImage=1, F=rl_F)
    l_lines = l_lines.reshape(-1, 3)

    l_points = np.cross(l_lines, l_line)  # crazy!
    valid = l_points[:, 2] != 0

    l_points = l_points[valid]
    r_points = r_points[valid]
    l_lines = l_lines[valid]

    l_points = l_points[:, :2] / l_points[:, 2][:, np.newaxis]
    valid = (l_points[:, 0] >= 0) & (l_points[:, 0] < n_cols) & (
        l_points[:, 1] >= 0) & (l_points[:, 1] < n_rows)
    l_points = l_points[valid]
    r_points = r_points[valid]
    l_lines = l_lines[valid]

    t_size = (80, 80)
    template = extract_window(f_image, f_point.astype(int), t_size)
    # template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    # l_gray = cv.cvtColor(l_image, cv.COLOR_BGR2GRAY)
    # r_gray = cv.cvtColor(r_image, cv.COLOR_BGR2GRAY)
    l_result, l_included = match_template(l_image, l_points, template, TM_SQDIFF_NORMED, np.inf)
    r_result, r_included = match_template(r_image, r_points, template, TM_SQDIFF_NORMED, np.inf)
    m_included = l_included & r_included
    m_result = 0.5 * np.array(l_result[m_included]) + 0.5 * np.array(r_result[m_included])

    if m_result.size == 0:
        return None

    r_point_match = r_points[m_included][np.argmin(m_result)]
    l_point_match = l_points[m_included][np.argmin(m_result)]
    l_line_match = l_lines[m_included][np.argmin(m_result)]

    return r_line, l_line, r_point_match, l_point_match, l_line_match


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--source-set', type=str, required=True)

    args = parser.parse_args()

    src_set = Path(abspath(args.source_set))

    # Load calibration targets
    print("[>] loading calibration targets...")
    l_calib_files = sorted(Path(f"{src_set}/calib/left").glob('*.JPG'))
    f_calib_files = sorted(Path(f"{src_set}/calib/front").glob('*.JPG'))
    r_calib_files = sorted(Path(f"{src_set}/calib/right").glob('*.JPG'))

    l_calib_images = [cv.resize(cv.imread(str(file)), TARGET_SIZE) for file in l_calib_files]
    f_calib_images = [cv.resize(cv.imread(str(file)), TARGET_SIZE) for file in f_calib_files]
    r_calib_images = [cv.resize(cv.imread(str(file)), TARGET_SIZE) for file in r_calib_files]

    # Define a dectector for ChArUco boards
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_50)
    board = cv.aruco.CharucoBoard(
        BOARD_SQ_SIZE, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    charuco_params = cv.aruco.CharucoParameters()
    charuco_params.tryRefineMarkers = False
    detector_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.CharucoDetector(board, charuco_params, detector_params)

    # Match calibration patterns for front and right cameras
    print("[>] matching calib patterns for front and right cameras...")
    try:
        obj_p_sets, f_img_p_sets, r_img_p_sets, f_corner_sets, r_corner_sets, f_calib_images_included, r_calib_images_included = match_and_show_calib_patterns(
            f_calib_images, r_calib_images, detector)
    except (RuntimeError, InterruptedError) as e:
        print(f"[e] {e}")
        exit(1)

    # Calibrate front and right cameras
    print("[>] calibrating front and right cameras...")
    try:
        fr_K, rf_K, error, fr_R, fr_t, fr_E, fr_F = calibrate_and_show_stereo(
            obj_p_sets, f_img_p_sets, r_img_p_sets, f_corner_sets, r_corner_sets, 
            f_calib_images_included, r_calib_images_included)
    except InterruptedError as e:
        print(f"[e] {e}")
        exit(1)

    print(f"[>] re-projection error front and right stereo: {error}")

    # Match calibration patterns for front and left cameras
    print("[>] matching calib patterns for front and left cameras...")
    try:
        obj_p_sets, f_img_p_sets, l_img_p_sets, f_corner_sets, l_corner_sets, f_calib_images_included, l_calib_images_included = match_and_show_calib_patterns(
            f_calib_images, l_calib_images, detector, flipped=True)
    except (RuntimeError, InterruptedError) as e:
        print(f"[e] {e}")
        exit(1)

    # Calibrate front and left cameras
    print("[>] calibrating front and left cameras...")
    try:
        fl_K, lf_K, error, fl_R, fl_t, fl_E, fl_F = calibrate_and_show_stereo(
            obj_p_sets, f_img_p_sets, l_img_p_sets, f_corner_sets, l_corner_sets, 
            f_calib_images_included, l_calib_images_included, flipped=True)
    except InterruptedError as e:
        print(f"[e] {e}")
        exit(1)

    print(f"[>] re-projection error front and left stereo: {error}")

    # Match calibration patterns for left and right cameras
    print("[>] matching calib patterns for left and right cameras...")
    try:
        obj_p_sets, l_img_p_sets, r_img_p_sets, l_corner_sets, r_corner_sets, l_calib_images_included, r_calib_images_included = match_and_show_calib_patterns(
            l_calib_images, r_calib_images, detector)
    except (RuntimeError, InterruptedError) as e:
        print(f"[e] {e}")
        exit(1)

    # Calibrate left and right cameras
    print("[>] calibrating left and right cameras...")
    try:
        _, _, error, rl_R, rl_t, rl_E, rl_F = calibrate_and_show_stereo(
            obj_p_sets, r_img_p_sets, l_img_p_sets, r_corner_sets, l_corner_sets, 
            r_calib_images_included, l_calib_images_included, flipped=True)
    except InterruptedError as e:
        print(f"[e] {e}")
        exit(1)

    print(f"[>] re-projection error left and right stereo: {error}")

    f_image = cv.resize(cv.imread(str(src_set/"front.JPG")), TARGET_SIZE)
    l_image = cv.resize(cv.imread(str(src_set/"left.JPG")), TARGET_SIZE)
    r_image = cv.resize(cv.imread(str(src_set/"right.JPG")), TARGET_SIZE)

    cv.imshow('right', cv.resize(r_image, VISUAL_SIZE))
    cv.imshow('left',  cv.resize(l_image, VISUAL_SIZE))
    cv.imshow('front', cv.resize(f_image, VISUAL_SIZE))

    fr_P = np.hstack([fr_K, np.zeros((3, 1))])
    rf_P = np.hstack([rf_K, np.zeros((3, 1))]) @ np.vstack([np.hstack([fr_R, fr_t]), np.array([0, 0, 0, 1])])

    fl_P = np.hstack([fl_K, np.zeros((3, 1))])
    lf_P = np.hstack([lf_K, np.zeros((3, 1))]) @ np.vstack([np.hstack([fl_R, fl_t]), np.array([0, 0, 0, 1])])

    point = np.array([np.nan, np.nan, np.nan])
    color = np.array([np.nan, np.nan, np.nan])
    param = ('front', f_image, 'right', r_image, 'left', l_image, fr_F, fl_F, rl_F, fl_P, lf_P, point, color)
    cv.setMouseCallback('front', trifocal_callback, param)

    # Reconstruct a sparse point cloud
    points = []
    colors = []

    while cv.getWindowProperty('front', cv.WND_PROP_VISIBLE) > 0:
        key = cv.waitKey(1)
        if key == 27:    # ESC key
            cv.destroyAllWindows()
            print("[e] ESC pressed")
            exit(1)
        elif key == 13:  # ENTER key
            cv.destroyAllWindows()
            break
        elif key == 32:  # SPACE key
            if not np.isnan(point).any() and not np.isnan(color).any():
                points.append(np.copy(point))
                colors.append(np.copy(color))
                print(f"[>] pushed vertex {point}")

    # n_cols, n_rows = TARGET_SIZE
    # for f_u in tqdm(range(80, n_cols - 80, 4)):
    #     for f_v in range(80, n_rows - 80, 4):
    #         f_point = np.array([f_u, f_v])
    #         match = match_trifocal(f_point, f_image, r_image, l_image, fr_F, fl_F, rl_F)
    #         if match is None:
    #             continue
        
    #         _, _, r_point_match, l_point_match, l_line_match = match
    #         t_point = cv.triangulatePoints(
    #             fl_P, lf_P, f_point.reshape(2, 1), l_point_match.reshape(2, 1))
    #         t_point = t_point.flatten()
    #         if t_point[3] < 0:
    #             continue

    #         point = t_point[0:3] / t_point[3]
    #         color = np.flip(f_image[int(f_v), int(f_u), :])

    #         points.append(point)
    #         colors.append(color)

    if len(points) > 0:
        points = np.array(points).reshape(-1, 3)
        colors = np.array(colors).reshape(-1, 3) / 255.

        np.save("out/points.npy", points)
        np.save("out/colors.npy", colors)
        
        pcloud = o3d.geometry.PointCloud()
        pcloud.points = o3d.utility.Vector3dVector(points)
        pcloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud('out/scloud.ply', pcloud)
        print("[>] saved sparse point clound")

    # Reconstruct local dense point clouds
    id = 0
    while True:
        print("[>] select a region of interest...")

        try:
            coords = prompt_image_point(cv.resize(f_image, VISUAL_SIZE), p_radius=5)

            # Ensure valid coords
            if not is_convex(coords):
                print("[!] ROI must be convex!")
                continue
            try:
                is_ccw(coords)
            except ValueError as e:
                print(f"[!] ROI is invalid (area={e})!")
                continue
            if is_ccw(coords):
                coords = np.flip(coords, axis=0)

            n_cols, _ = TARGET_SIZE
            v_cols, _ = VISUAL_SIZE

            pts = coords * ((n_cols - 1.) / (v_cols - 1.))
            roi = build_mask(f_image.shape, pts)
            uvs = np.column_stack(np.where(roi))

            step = 4
            uvs = uvs[::step]

            points = []
            included = np.ones(len(uvs), dtype=bool)
            for i, (v, u) in enumerate(tqdm(uvs)):
                f_point = np.array([u, v])
                match = match_trifocal(f_point, f_image, r_image, l_image, fr_F, fl_F, rl_F)

                if match is None:
                    included[i] = False
                    continue

                _, _, r_point_match, l_point_match, _ = match
                t_point = cv.triangulatePoints(
                    fl_P, lf_P, f_point.reshape(2, 1).astype(float), 
                    l_point_match.reshape(2, 1).astype(float))
                t_point = t_point.flatten()
                if t_point[3] < 0:
                    included[i] = False
                    continue

                point = t_point[0:3] / t_point[3]
                points.append(point)

            colors = np.flip(f_image[roi][::step][included], axis=1)

            points = np.array(points).reshape(-1, 3)
            colors = np.array(colors).reshape(-1, 3) / 255.

            np.save(f"out/points_{id}.npy", points)
            np.save(f"out/colors_{id}.npy", colors)

            pcloud = o3d.geometry.PointCloud()
            pcloud.points = o3d.utility.Vector3dVector(points)
            pcloud.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(f"out/cloud_{id}.ply", pcloud)
            print(f"[>] saved local dense point clound {id}")

            id = id + 1
        except InterruptedError:
            break
