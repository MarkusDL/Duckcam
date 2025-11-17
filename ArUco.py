import cv2
import cv2.aruco as aruco
import numpy as np
from collections import defaultdict

# Load the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

def get_angle(corners):
    """Compute orientation angle of a marker from its first two corners."""
    v = np.array(corners[1]) - np.array(corners[0])
    return np.degrees(np.arctan2(v[1], v[0]))

def is_perpendicular(angle1, angle2, tol=15):
    """Check if two angles are approximately perpendicular."""
    diff = abs(angle1 - angle2) % 180
    return abs(diff - 90) < tol

def approximate_square(center, side_len=600):
    """Approximate a square centered at given point."""
    cx, cy = center
    return [
        (cx - side_len/2, cy - side_len/2),
        (cx + side_len/2, cy - side_len/2),
        (cx + side_len/2, cy + side_len/2),
        (cx - side_len/2, cy + side_len/2)
    ]

def build_square_two_markers(m1, m2):
    """Build square from two markers."""
    c1, c2 = np.array(m1["center"]), np.array(m2["center"])
    angle1, angle2 = get_angle(m1["corners"]), get_angle(m2["corners"])
    dist = np.linalg.norm(c2 - c1)

    # Midpoint and orientation
    midpoint = (c1 + c2) / 2
    if is_perpendicular(angle1, angle2):
        # Adjacent sides: use dist as side length
        side_len = dist
    else:
        # Opposite sides: use dist as diagonal
        side_len = dist / np.sqrt(2)

    # Compute square corners aligned with angle1
    theta = np.radians(angle1)
    dx, dy = np.cos(theta), np.sin(theta)
    half = side_len / 2
    # Rotate 90° for perpendicular axis
    perp_dx, perp_dy = -dy, dx

    return [
        (midpoint[0] - dx*half - perp_dx*half, midpoint[1] - dy*half - perp_dy*half),
        (midpoint[0] + dx*half - perp_dx*half, midpoint[1] + dy*half - perp_dy*half),
        (midpoint[0] + dx*half + perp_dx*half, midpoint[1] + dy*half + perp_dy*half),
        (midpoint[0] - dx*half + perp_dx*half, midpoint[1] - dy*half + perp_dy*half)
    ]

def infer_squares(final_markers, default_side_len=50):
    """Infer squares for each marker ID from 1–4 markers."""
    marker_groups = defaultdict(list)
    for m in final_markers:
        marker_groups[m["id"]].append(m)

    squares = {}
    for marker_id, group in marker_groups.items():
        if len(group) == 1:
            # Approximate square
            squares[marker_id] = approximate_square(group[0]["center"], default_side_len)
        elif len(group) == 2:
            squares[marker_id] = build_square_two_markers(group[0], group[1])
        elif len(group) >= 3:
            # Use convex hull of centers for 3 or 4 markers
            pts = np.array([m["center"] for m in group])
            hull = pts[np.argsort(np.arctan2(pts[:,1]-pts[:,1].mean(), pts[:,0]-pts[:,0].mean()))]
            squares[marker_id] = hull.tolist()
        # If more than 4 markers, still return hull
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
    marker_list: list of dicts with 'id' and 'corners'
    """
    deduped = {}
    
    for marker in marker_list:
        marker_id = marker["id"]
        corners = np.array(marker["corners"])
        center = np.array(marker["center"])

        if marker_id not in deduped:
            deduped[marker_id] = [{"center": center, "corners": corners}]
        else:
            # Check if this marker is close to any existing one
            merged = False
            for existing in deduped[marker_id]:
                dist = np.linalg.norm(center - existing["center"])
                if dist < distance_threshold:
                    # Merge by averaging
                    existing["center"] = (existing["center"] + center) / 2
                    existing["corners"] = (existing["corners"] + corners) / 2
                    merged = True
                    break
            if not merged:
                deduped[marker_id].append({"center": center, "corners": corners})

    # Flatten and format output
    result = []
    for marker_id, instances in deduped.items():
        for inst in instances:
            result.append({
                "id": marker_id,
                "center": inst["center"].tolist(),
                "corners": inst["corners"].tolist()
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
