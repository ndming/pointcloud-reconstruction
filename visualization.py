import cv2 as cv
import matplotlib.pyplot as plt
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


def visualize_charuco_detection(image, corners, radius=10):
    # Draw points on the image
    draw_img = np.copy(image)

    n_colors = len(_COLORS)
    n_points = corners.shape[0]

    for corner, i in zip(corners, range(n_points)):
        corner = corner.astype(int)
        cv.circle(draw_img, corner, radius, _COLORS[i % n_colors], -1)

    # height, width, _ = draw_img.shape
    # clip_width = width // 4  # 50% into each dimension
    # clip_height = height // 4  # 50% into each dimension

    # Define the coordinates of the top-left and bottom-right corners of the 
    # clipped portion
    # x1 = clip_width
    # y1 = clip_height
    # x2 = width - clip_width
    # y2 = height - clip_height

    # Clip the portion of the image defined by the calculated dimensions
    # draw_img = draw_img[y1:y2, x1:x2]

    _, axes = plt.subplots(1, 2)

    # Display the images in the subplots
    axes[0].imshow(np.flip(image, axis=2))
    axes[0].set_title('Original image')

    axes[1].imshow(np.flip(draw_img, axis=2))
    axes[1].set_title('Detection of ChArUco corners')

    # Hide the axes
    for ax in axes:
        ax.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()


def visualize_epipolar_lines(
    img_1, img_2, points_1, lines_2, radius=20, thickness=20):
    n_rows, n_cols, _ = img_1.shape
    n_colors = len(_COLORS)
    n_points = points_1.shape[0]

    # First, draw points on the first image
    draw_img_1 = np.copy(img_1)
    for point, i in zip(points_1, range(n_points)):
        point = point.astype(int)
        color = _COLORS[i % n_colors]
        cv.circle(
            draw_img_1, point, radius=radius, color=color, thickness=-1)
        
    # Then draw their corresponding lines on the second image
    draw_img_2 = np.copy(img_2)
    for line, i in zip(lines_2, range(n_points)):
        x0, y0 = map(int, [0, -line[2] / line[1] ])
        x1 ,y1 = map(int, [n_cols, -(line[2] + line[0] * n_cols) / line[1]])
        color = _COLORS[i % n_colors]
        cv.line(draw_img_2, (x0, y0), (x1, y1), color, thickness)

    _, axes = plt.subplots(1, 2)

    # Display the images in the subplots
    axes[0].imshow(np.flip(draw_img_1, axis=2))
    axes[0].set_title('Points in the 1st image')

    axes[1].imshow(np.flip(draw_img_2, axis=2))
    axes[1].set_title('Their epi-lines in the 2nd image')

    # Hide the axes
    for ax in axes:
        ax.axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()