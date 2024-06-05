import cv2 as cv
import numpy as np


_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (128, 0, 0),      # Maroon
    (0, 128, 0),      # Dark Green
    (0, 0, 128),      # Navy
    (128, 128, 0),    # Olive
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
    (192, 192, 192),  # Silver
    (128, 128, 128),  # Gray
    (255, 165, 0),    # Orange
    (255, 20, 147),   # Deep Pink
    (75, 0, 130),     # Indigo
    (255, 192, 203),  # Pink
]


def draw_stereo_matching(
    images_1: list[np.array],
    images_2: list[np.array], 
    corner_sets_1: list[np.array], 
    corner_sets_2: list[np.array],
    radius=30
) -> tuple[list[np.array], list[np.array]]:
    drawings_1 = []
    drawings_2 = []

    iterable = zip(images_1, images_2, corner_sets_1, corner_sets_2)
    for image_1, image_2, corners_1, corners_2 in iterable:
        n_colors = len(_COLORS)

        drawing_1 = np.copy(image_1)
        drawing_2 = np.copy(image_2)

        for i, corner in enumerate(corners_1):
            point = corner.astype(int)
            cv.circle(drawing_1, point, radius, _COLORS[i % n_colors], -1)

        for i, corner in enumerate(corners_2):
            point = corner.astype(int)
            cv.circle(drawing_2, point, radius, _COLORS[i % n_colors], -1)

        drawings_1.append(drawing_1)
        drawings_2.append(drawing_2)

    return drawings_1, drawings_2


def draw_stereo_epilines(
    images_1: list[np.array], 
    images_2: list[np.array], 
    corner_sets_1:  list[np.array], 
    epiline_sets_2: list[np.array], 
    radius=20,
    thickness=16
) -> tuple[list[np.array], list[np.array]]:
    _, n_cols, _ = images_1[0].shape
    n_colors = len(_COLORS)

    drawings_1 = []
    drawings_2 = []

    iterable = zip(images_1, images_2, corner_sets_1, epiline_sets_2)
    for image_1, image_2, points_1, lines_2 in iterable:
        drawing_1 = np.copy(image_1)
        drawing_2 = np.copy(image_2)

        for i, point in enumerate(points_1):
            point = point.astype(int)
            color = _COLORS[i % n_colors]
            cv.circle(drawing_1, point, radius, color, thickness=-1)

        for i, line in enumerate(lines_2):
            x0, y0 = map(int, [0, -line[2] / line[1] ])
            x1 ,y1 = map(int, [n_cols, -(line[2] + line[0] * n_cols) / line[1]])
            color = _COLORS[i % n_colors]
            cv.line(drawing_2, (x0, y0), (x1, y1), color, thickness)

        drawings_1.append(drawing_1)
        drawings_2.append(drawing_2)

    return drawings_1, drawings_2


def draw_lines(image, lines, color, thickness=5):
    _, n_cols = image.shape[:2]
    for line in lines:
        x_0, y_0 = map(int, [0, -line[2] / line[1]])
        x_1, y_1 = map(int, [n_cols, -(line[2] + line[0] * n_cols) / line[1]])
        cv.line(image, (x_0, y_0), (x_1, y_1), color, thickness)