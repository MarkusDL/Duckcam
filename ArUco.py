import cv2
import cv2.aruco as aruco
import numpy as np
from collections import defaultdict

# Load the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
import numpy as np
import cv2


def order_points(points):
    # points: list of 4 np.array([x, y, z])
    center = np.mean(points, axis=0)
    
    def angle(p):
        vec = p - center
        return np.arctan2(vec[1], vec[0])  # XY plane
    
    return sorted(points, key=angle)

    
def project_square_to_image(square_3d, camera_matrix, dist_coeffs):
    pts = np.array(square_3d, dtype=np.float32)
    img_pts, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    return img_pts.reshape(-1, 2).tolist()

        
def get_rotation_matrix(rvec):
    R, _ = cv2.Rodrigues(np.array(rvec))
    return R

def build_square_from_axes(origin, x_axis, y_axis, size_x, size_y):
    """Build square corners in 3D given origin and axes."""
    return [
        origin.tolist(),
        (origin + x_axis * size_x).tolist(),
        (origin + x_axis * size_x + y_axis * size_y).tolist(),
        (origin + y_axis * size_y).tolist()
    ]

def infer_square_for_group(markers, default_edge=0.550):
    """
    Infer square for a group of markers with same ID.
    Handles 1–4 markers.
    """
    poses = []
    for m in markers:
        tvec = np.array(m["t_vec"]).flatten()
        rvec = np.array(m["r_vec"]).flatten()
        R = get_rotation_matrix(rvec)
        poses.append({"pos": tvec, "R": R})

    if len(poses) == 1:
        print("found square with only one marker")
        # Case 1: Single marker
        origin = poses[0]["pos"]
        x_axis = poses[0]["R"][:, 0]
        # Use marker's Y as your X
        y_axis = poses[0]["R"][:, 1]
        # Use marker's X as your Y
        return build_square_from_axes(origin, x_axis, y_axis, -default_edge, -default_edge)

    elif len(poses) == 2:
        print("found square with only two markers")
        # Case 2: Two markers
        p1, p2 = poses[0]["pos"], poses[1]["pos"]
        R1, R2 = poses[0]["R"], poses[1]["R"]
        dist = np.linalg.norm(p2 - p1)

        # Check orientation difference
        dot_x = np.dot(R1[:, 0], R2[:, 0])
        dot_y = np.dot(R1[:, 1], R2[:, 1])

        origin = p1
        # Use marker's Y as your X
        x_axis = R1[:, 1]
        # Use marker's X as your Y
        y_axis = R1[:, 0]

        if abs(dot_x) > 0.9:  # Same edge
            return build_square_from_axes(origin, x_axis, y_axis, dist, default_edge)
        else:  # Perpendicular
            return build_square_from_axes(origin, x_axis, y_axis, dist, dist)

    elif len(poses) == 3:
        positions = [p["pos"] for p in poses]
        p1, p2, p3 = positions
    
        # Compute vectors
        v1 = p2 - p1
        v2 = p3 - p1
    
        # Normalize to same length
        side_len = min(np.linalg.norm(v1), np.linalg.norm(v2))
        v1 = v1 / np.linalg.norm(v1) * side_len
        v2 = v2 / np.linalg.norm(v2) * side_len
    
        # Compute fourth point
        p4 = p1 + v1 + v2

        return order_points([p1, p2, p3, p4])

    else:
        # Case 4: Four markers → fully constrained
        positions = [p["pos"].tolist() for p in poses]
        R = poses[0]["R"]
        x_axis, y_axis = R[:, 0], R[:, 1]

        # Compute extents
        #origin = min(positions, key=lambda p: np.dot(p, x_axis) + np.dot(p, y_axis))
        #max_x = max(np.dot(p - origin, x_axis) for p in positions)
        #max_y = max(np.dot(p - origin, y_axis) for p in positions)
        return positions

def infer_squares(markers, default_edge=0.6):
    """
    Group markers by ID and infer squares.
    """
    groups = {}
    for m in markers:
        groups.setdefault(m["id"], []).append(m)

    squares = {}
    for marker_id, group in groups.items():
        square_3d = infer_square_for_group(group, default_edge)
        squares[marker_id] = {"square_3d": square_3d}
    return squares

def get_marker_center(corners):
    """Calculate the center of the marker from its corners."""
    return np.mean(corners, axis=0)

def detect_aruco_markers(image):

    image = np.array(image, copy=False)

    # Convert 4-channel to 3-channel if needed
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Validate format
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Invalid image format for ArUco detection: shape={image.shape}, dtype={image.dtype}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    return ids, corners

def split_into_tiles(image, tile_size=(1000, 1000), overlap=0.5):
    tiles = []
    step_x = int(tile_size[0] * (1 - overlap))
    step_y = int(tile_size[1] * (1 - overlap))
    h, w = image.shape[:2]

    for y in range(0, h - tile_size[1] + 1, step_y):
        for x in range(0, w - tile_size[0] + 1, step_x):
            tile = image[y:y + tile_size[1], x:x + tile_size[0]]
            tiles.append((tile, x, y))
    return tiles

import numpy as np



def deduplicate_markers(marker_list, distance_threshold=20):
    """
    Deduplicate markers based on ID and proximity.
    Preserves r_vec and t_vec for pose-based calculations.
    """
    deduped = {}

    for marker in marker_list:
        marker_id = marker["id"]
        center = np.array(marker["center"])
        corners = np.array(marker["corners"])
        r_vec = np.array(marker["r_vec"])
        t_vec = np.array(marker["t_vec"])

        if marker_id not in deduped:
            deduped[marker_id] = [{
                "center": center,
                "corners": corners,
                "r_vec": r_vec,
                "t_vec": t_vec
            }]
        else:
            merged = False
            for existing in deduped[marker_id]:
                dist = np.linalg.norm(center - existing["center"])
                if dist < distance_threshold:
                    # Merge by averaging pose and geometry
                    existing["center"] = (existing["center"] + center) / 2
                    existing["corners"] = (existing["corners"] + corners) / 2
                    existing["r_vec"] = (existing["r_vec"] + r_vec) / 2
                    existing["t_vec"] = (existing["t_vec"] + t_vec) / 2
                    merged = True
                    break
            if not merged:
                deduped[marker_id].append({
                    "center": center,
                    "corners": corners,
                    "r_vec": r_vec,
                    "t_vec": t_vec
                })

    # Flatten and format output
    result = []
    for marker_id, instances in deduped.items():
        for inst in instances:
            result.append({
                "id": marker_id,
                "center": inst["center"].tolist(),
                "corners": inst["corners"].tolist(),
                "r_vec": inst["r_vec"].tolist(),
                "t_vec": inst["t_vec"].tolist()
            })

    return result

def load_and_detect(path):
    """
    Loads an image from disk and detects ArUco markers.

    Args:
        path (str): Path to the image file.

    Returns:
        ids (np.ndarray): Array of detected marker IDs.
    """
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not load image from path: {path}")

    corners, ids, image_with_markers = detect_aruco_markers(image)

    # Optionally show the result
    cv2.imshow("Detected ArUco Markers", image_with_markers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return ids
