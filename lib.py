import os
import cv2 as cv
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
from skimage.morphology import binary_closing
import logging
import sys

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))

# not used in main program, but useful for tests
def read_images(dir: str) -> list:
    images = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            images.append(cv.imread(os.path.join(dirpath, filename)))
    return images


def read_image(path: str) -> list:
    return cv.imread(path)


# find edges with canny detection and morphology operations
def preprocess_image(image: np.array) -> np.array:
    gray_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edge = 255 * binary_fill_holes(binary_closing(canny(gray_img, sigma=3), footprint=np.ones((20, 20))))
    edge = edge.astype(np.uint8)
    return edge


# is contour a polygon
def check_for_polygon(contours: list) -> bool:
    counter = 0
    for contour in contours:
        x, y = contour[0]
        if y < 750:
            counter += 1
    if counter == len(contours):
        return True
    return False


# is contour a bottom paper sheet border
def check_paper_border(contours: list) -> bool:
    paper_border_counter = 0
    for contour in contours:
        x, y = contour[0]
        if 750 > y > 600:
            paper_border_counter += 1
            continue
    if paper_border_counter == len(contours):
        return False
    return True


# finding a minimal rectangle for objects contours
def find_min_rect(contours: list, image: np.array=None) -> list:
    boxes = list()
    for c in contours:
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)
    return boxes


# find polygon angles
def find_approx_polygon(polygon_cnt: list) -> list:
    epsilon = 0.01 * cv.arcLength(polygon_cnt, True)
    approx = cv.approxPolyDP(polygon_cnt, epsilon, True)
    return approx


# find contours of polygon and all objects
def find_contours(image: np.array) -> tuple:
    contours = cv.findContours(image, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    objects = list()
    polygon = None
    for contour in contours[0]:
        if check_for_polygon(contour) and cv.contourArea(contour) > 3000 and check_paper_border(contour):
            polygon = contour
        elif not check_for_polygon(contour) and cv.contourArea(contour) > 500:
            objects.append(contour)
    return polygon, objects

def check_area(polygon: list, objects: list) -> bool:
    poly_area = cv.contourArea(polygon)
    obj_area = 0
    for obj in objects:
        obj_area += cv.contourArea(obj)
    return poly_area > obj_area


def is_objects_placed(polygon: list, objects: list) -> bool:
    if check_area(polygon, objects):
        return True
    else:
        return False


def check_image(path: str) -> bool:
    image = read_image(path)
    edge = preprocess_image(image)
    polygon, objects = find_contours(edge)
    if polygon is None or objects == []:
        print("Can't find polygon or objects")
        return False
    rectangles = find_min_rect(objects)
    polygon_angles = find_approx_polygon(polygon)
    return is_objects_placed(polygon_angles, rectangles)
