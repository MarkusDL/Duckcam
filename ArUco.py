import cv2
import cv2.aruco as aruco
import numpy as np
from collections import defaultdict

# Load the ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

def get_local_axes(corners):
    """Compute local X and Y axes from marker corners."""
    if not corners or len(corners) < 2:
        raise ValueError(f"Invalid corners: {corners}")
    # Normalize format: if nested [[[x,y],...]], flatten it
    if isinstance(corners[0][0], (list, tuple)):
        corners = corners[0]
    p0, p1 = np.array(corners[0]), np.array(corners[1])
    x_axis = p1 - p0
    norm = np.linalg.norm(x_axis)
    if norm == 0:
        raise ValueError("Zero-length axis vector")
    x_axis /= norm
    y_axis = np.array([-x_axis[1], x_axis[0]])  # rotate 90°
    return x_axis, y_axis



def build_square_one_marker(marker, side_len=600):
    """Square with marker as one corner, aligned to marker axes."""
    center = np.array(marker["center"])
    x_axis, y_axis = get_local_axes(marker["corners"])
    return [
        center.tolist(),
        (center + x_axis * side_len).tolist(),
        (center + x_axis * side_len + y_axis * side_len).tolist(),
        (center + y_axis * side_len).tolist()
    ]

def build_square_two_markers(m1, m2):
    """Square with edge defined by two markers, aligned to m1 axes."""
    c1, c2 = np.array(m1["center"]), np.array(m2["center"])
    x_axis, y_axis = get_local_axes(m1["corners"])
    side_len = np.linalg.norm(c2 - c1)
    return [
        c1.tolist(),
        (c1 + x_axis * side_len).tolist(),
        (c1 + x_axis * side_len + y_axis * side_len).tolist(),
        (c1 + y_axis * side_len).tolist()
    ]



def build_square_three_or_four(markers):
    """Use first marker as reference, assume 90° rotations around square."""
    markers = sorted(markers, key=lambda m: m["id"])
    m1 = markers[0]
    c1 = np.array(m1["center"])
    x_axis, y_axis = get_local_axes(m1["corners"])
    # Estimate side length from nearest neighbor
    distances = [np.linalg.norm(np.array(m["center"]) - c1) for m in markers[1:]]
    side_len = min(distances) if distances else 50
    return [
        c1.tolist(),
        (c1 + x_axis * side_len).tolist(),
        (c1 + x_axis * side_len + y_axis * side_len).tolist(),
        (c1 + y_axis * side_len).tolist()
    ]



def infer_squares(final_markers, default_side_len=50):
    """Infer squares for each marker ID using local axes."""
    marker_groups = defaultdict(list)
    for m in final_markers:
        marker_groups[m["id"]].append(m)

    squares = {}
    for marker_id, group in marker_groups.items():
        # Filter out invalid markers
        valid_group = [m for m in group if m.get("corners") and len(m["corners"]) >= 2]
        if not valid_group:
            continue
        try:
            if len(valid_group) == 1:
                squares[marker_id] = build_square_one_marker(valid_group[0], default_side_len)
            elif len(valid_group) == 2:
                squares[marker_id] = build_square_two_markers(valid_group[0], valid_group[1])
            else:
                squares[marker_id] = build_square_three_or_four(valid_group)
        except ValueError as e:
            print(f"Skipping marker {marker_id}: {e}")
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
