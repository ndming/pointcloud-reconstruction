import cv2 as cv
import numpy as np


def prompt_image_point(image, win_name="image_front", p_radius=5):
    img = np.copy(image)
    points = []
    param = (img, points, win_name, p_radius)

    cv.imshow(win_name, img)
    cv.setMouseCallback(win_name, _select_point_callback, param)

    while cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) > 0:
        key = cv.waitKey(1)
        if key == 27:    # ESC key
            cv.destroyAllWindows()
            raise InterruptedError()
        elif key == 13:  # ENTER key
            if len(points) < 3:
                print("[!] ROI must consist of > 3 points")
                continue

            cv.destroyAllWindows()
            return np.array(points).astype(float)
        elif key == 32:  # SPACE key
            points.clear()

            img = np.copy(image)
            param = (img, points, win_name, p_radius)
            
            cv.imshow(win_name, img)
            cv.setMouseCallback(win_name, _select_point_callback, param)


def _select_point_callback(event, x, y, _, param):
    if event == cv.EVENT_LBUTTONDOWN:
        img, points, win_name, radius = param
        points.append((x, y))
        cv.circle(img, (x, y), radius, (0, 255, 0), -1)
        cv.imshow(win_name, img)


def build_mask(
    mask_shape: np.ndarray | tuple[int, int], 
    pln_coords: np.ndarray
) -> np.ndarray[bool]:
    """
    Generates a 2D boolean mask of size `mask_shape` where pixels within the 
    polygon defined by the coordinate sequence `pln_coords` are assigned `True`, 
    whereas pixels outside are assigned `False`. 
    
    The coordinate sequence must define a convex polygon in CCW order, without 
    duplicate for the first and end point.

    Parameters:
    ---
    - `mask_shape`: the size of the output mask defined as (`n_rows`, `n_cols`)
    - `pln_coords`: an `n-by-2` array containing the image coordinates defining 
    the convex polygon in CCW order

    Returns:
    ---
    A boolean mask where `True` indicates coordinates within the defined polygon 
    and `False` indicates coordinates outside.

    Raises:
    ---
    - `ValueError`: if the number of coordinates is less than 3 or the input 
    `pln_coords` array is not valid.

    Example:
    ---
    >>> # Define polygon vertices
    >>> polygon_vertices = np.array([[100, 50], [200, 100], [150, 200]])

    >>> # Define the shape of the output mask
    >>> output_shape = (300, 300)

    >>> # Generate boolean mask for the polygon
    >>> mask = build_mask(output_shape, polygon_vertices)
    """

    n_points = pln_coords.shape[0]
    if n_points < 3 or np.ndim(pln_coords) < 2:
        raise ValueError(f"pln_coords array is not valid")
    
    # Define an array of adjacent points
    adj_coords = np.zeros(pln_coords.shape)
    adj_coords[0:-1] = pln_coords[1:]
    adj_coords[-1] = pln_coords[0]

    # The normal vectors for each line segmentS
    directions = adj_coords - pln_coords
    normals = directions[:, [1, 0]] * np.array([1, -1])

    rows, cols = mask_shape[:2]
    m_x, m_y = np.meshgrid(np.arange(cols), np.arange(rows))
    grid = np.vstack([m_x.flatten(), m_y.flatten()]).T
    grid = np.tile(np.expand_dims(grid, axis=2), [1, 1, n_points])
    normals = np.swapaxes(np.expand_dims(normals, axis=2), 0, 2)
    coords = np.swapaxes(np.expand_dims(pln_coords, axis=2), 0, 2)

    # Check if points in the mesh grid lie on the left or on the right to each
    # line segment, then "and" all results
    mask = np.sum((grid - coords) * normals, axis=1) >= 0
    mask = np.logical_and.reduce(mask, axis=1).reshape(rows, cols)

    return mask