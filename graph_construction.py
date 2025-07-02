import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy.spatial import KDTree

def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (256, 128))
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    return skeletonize(thresh // 255)

def extract_keypoints(skeleton):
    keypoints = []
    h, w = skeleton.shape
    for y in range(h):
        for x in range(w):
            if skeleton[y, x] == 1:
                neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 1
                if neighbors == 1 or neighbors >= 3:
                    keypoints.append((x, y))
    return keypoints

def build_graph(keypoints, max_dist=10, k=5):
    tree = KDTree(keypoints)
    edges = []
    for i, point in enumerate(keypoints):
        dist, idxs = tree.query(point, k=k+1)
        for d, j in zip(dist[1:], idxs[1:]):
            if d <= max_dist:
                edges.append((i, j))
    return nx.Graph(edges), keypoints
