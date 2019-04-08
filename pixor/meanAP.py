import cv2
import math
import numpy as np


#taken from nms package 
def poly_areas(polys):
    """Calculate the area of the list of polygons

    :param polys: a list of polygons, each specified by a list of its verticies
    :type polys: list
    :return: numpy array of areas of the polygons
    :rtype: :class:`numpy.ndarray`
    """
    areas = []
    for poly in polys:
        areas.append(cv2.contourArea(np.array(poly, np.int32)))
    return np.array(areas)

def extract_unique_labels(boxes):
    unique_boxes_set = set()
    unique_boxes = {}
    for r in range(0, boxes.shape[0]):
        for c in range(0, boxes.shape[1]):
            if tuple(boxes[r,c][2:]) not in unique_boxes_set:
                unique_boxes_set.add(tuple(boxes[r,c][2:]))
                center_x = (c) + (int(boxes[r,c][0]))
                center_y = (r) + (int(boxes[r,c][1]))
                center = np.array([center_x, center_y])
                others = np.array([boxes[r,c][2], boxes[r,c][3], boxes[r,c][4], boxes[r,c][5]])
                print("others")
                print(others)
                print(boxes[r,c][2:])
                box = np.concatenate([center, others])
                unique_boxes[str(r) + "_" + str(c)] = box
    return list(unique_boxes.values())

#taken from nms package
def createImage(width=800, height=800, depth=3):
    """ Return a black image with an optional scale on the edge

    :param width: width of the returned image
    :type width: int
    :param height: height of the returned image
    :type height: int
    :param depth: either 3 (rgb/bgr) or 1 (mono).  If 1, no scale is drawn
    :type depth: int
    :return: A zero'd out matrix/black image of size (width, height)
    :rtype: :class:`numpy.ndarray`
    """
    # create a black image and put a scale on the edge

    assert depth == 3 or depth == 1
    assert width > 0
    assert height > 0

    hashDistance = 50
    hashLength = 20

    img = np.zeros((int(height), int(width), depth), np.uint8)

    if(depth == 3):
        for x in range(0, int(width / hashDistance)):
            cv2.line(img, (x * hashDistance, 0), (x * hashDistance, hashLength), (0,0,255), 1)

        for y in range(0, int(width / hashDistance)):
            cv2.line(img, (0, y * hashDistance), (hashLength, y * hashDistance), (0,0,255), 1)

    return img

#from nms package
def polygon_intersection_area(polygons):
    """ Compute the area of intersection of an array of polygons

    :param polygons: a list of polygons
    :type polygons: list
    :return: the area of intersection of the polygons
    :rtype: int
    """
    if len(polygons) == 0:
        return 0

    dx = 0
    dy = 0

    maxx = np.amax(np.array(polygons)[...,0])
    minx = np.amin(np.array(polygons)[...,0])
    maxy = np.amax(np.array(polygons)[...,1])
    miny = np.amin(np.array(polygons)[...,1])

    if minx < 0:
        dx = -int(minx)
        maxx = maxx + dx
    if miny < 0:
        dy = -int(miny)
        maxy = maxy + dy
    # (dx, dy) is used as an offset in fillPoly

    for i, polypoints in enumerate(polygons):

        newImage = createImage(maxx, maxy, 1)

        polypoints = np.array(polypoints, np.int32)
        polypoints = polypoints.reshape(-1, 1, 2)

        cv2.fillPoly(newImage, [polypoints], (255, 255, 255), cv2.LINE_8, 0, (dx, dy))

        if(i == 0):
            compositeImage = newImage
        else:
            compositeImage = cv2.bitwise_and(compositeImage, newImage)

        area = cv2.countNonZero(compositeImage)

    return area

#from nms package
def poly_compare(poly1, polygons, truth_areas):
    """Calculate the intersection of poly1 to polygons divided by area

    :param poly1: a polygon specified by a list of its verticies
    :type poly1: list
    :param polygons: a list of polygons, each specified a list of its verticies
    :type polygons: list
    :param area: a list of areas of the corresponding polygons
    :type area: list
    :return: a numpy array of the ratio of overlap of poly1 to each of polygons to the corresponding area.  e.g. overlap(poly1, polygons[n])/area[n]
    :rtype: :class:`numpy.ndarray`
    """
    # return intersection of poly1 with polys[i]/area[i]
    overlap = []
    pred_area = cv2.contourArea(np.array(poly1, np.int32))
    for i,poly2 in enumerate(polygons):
        intersection_area = polygon_intersection_area([poly1, poly2])
        overlap.append(intersection_area/(truth_areas[i]+pred_area))

    return np.array(overlap)

def convert_to_poly(boxes):
    polys = []
    for box in boxes:
        r = cv2.boxPoints(box)
        if not np.isnan(r).any():
            polys.append(r)
        
    return polys

# Sort predicted boxes by confidence values
#Per image
## For each box:
#### calculate its IoU of every ground truth box
#### save counter of highest IoU
#### remove ground truth box
## If there's a match, tp++, else, fp++

## map = tp/(tp+fp)

#from nms package
def image_meanAP(predictions, truth, threshold):
    predictions = convert_to_poly(predictions)
    truth = convert_to_poly(truth)
    
    true_pos = 0.
    false_pos = 0.
    truth_areas = poly_areas(truth)
    for box in predictions:
        ratios = poly_compare(box, truth, truth_areas)
        best_index = np.argsort(ratios)[-1]
        best_IoU = ratios[best_index]
        print("best_IoU")
        print(best_IoU)
        if best_IoU >= threshold:
            true_pos += 1
            truth = np.delete(truth, best_index)
        else:
            false_pos += 1
    print("num of true pos")
    print(true_pos)
    print("num of false pos")
    print(false_pos)
    meanAP = (true_pos)/(true_pos+false_pos) if false_pos != 0. else 0  
        
    
    return meanAP