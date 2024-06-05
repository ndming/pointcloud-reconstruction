import numpy as np


def extract_window(image, point, size=(200, 200)):
    u, v = point
    w, h = size

    top_left_u = max(u - w // 2, 0)
    top_left_v = max(v - h // 2, 0)
    bottom_right_u = min(u + w // 2, image.shape[1])
    bottom_right_v = min(v + h // 2, image.shape[0])

    return image[top_left_v:bottom_right_v, top_left_u:bottom_right_u]