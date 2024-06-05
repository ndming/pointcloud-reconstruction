import numpy as np


def extract_window(image, point, size=(200, 200)):
    u, v = point
    w, h = size

    top_left_u = max(u - w // 2, 0)
    top_left_v = max(v - h // 2, 0)
    bottom_right_u = min(u + w // 2, image.shape[1])
    bottom_right_v = min(v + h // 2, image.shape[0])

    return image[top_left_v:bottom_right_v, top_left_u:bottom_right_u]


def match_template(image, points, template, method, fill_val):
    n_rows, n_cols = image.shape[:2]
    t_rows, t_cols = template.shape[:2]
    result = np.full(len(points), fill_val)
    included = np.ones(len(points), dtype=bool)
    for i, (u, v) in enumerate(points):
        if u < t_cols // 2 or u > n_cols - t_cols // 2 or v < t_rows // 2 or v > n_rows - t_rows // 2:
            included[i] = False
            continue

        window = extract_window(image, (int(u), int(v)), (t_cols, t_rows))
        if window.shape != template.shape:
            included[i] = False
            continue
        
        result[i] = method(window.astype(float), template.astype(float))

    return result, included


def TM_SQDIFF(window, template):
    return np.sum((window - template) ** 2)


def TM_SQDIFF_NORMED(window, template):
    num = np.sum((window - template) ** 2)
    den = np.sqrt(np.sum(window ** 2) * np.sum(template ** 2))
    return num / den


def TM_CCORR(window, template):
    return np.sum(window * template)


def TM_CCOEFF(window, template):
    w_mean = np.mean(window)
    t_mean = np.mean(template)
    num = np.sum((window - w_mean) * (template - t_mean))
    den = np.sqrt(np.sum((window - w_mean) ** 2) * np.sum((template - t_mean) ** 2))
    return num / den
    