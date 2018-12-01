from lxml import etree
import numpy as np
import scipy.misc
import pickle
from data_extract import extract_data
from random import shuffle


TRAIN_ANNOTATIONS_PATH = "./dataset/train/annotations/"
TRAIN_IMG_PATH = "./dataset/train/img/"

VAL_ANNOTATIONS_PATH = "./dataset/val/img/"
VAL_IMG_PATH = "./dataset/val/img/"

TEST_ANNOTATIONS_PATH = "./dataset/test/img/"
TEST_IMG_PATH = "./dataset/test/img/"



# Images and bboxes array -> write named images to folder, write xml annotations for each image to separate folder. 

def compile_dataset(path_to_data):
    data = extract_data(path_to_data)
    shuffle(data)

    # data is an array of tuples. Each tuple contains an image arrays and a bbox array
    for elem in enumerate(data):
        # elem is (int index of tuple, tuple)

        # index of (tile, bboxes) in data
        im_num = elem[0]

        img_name = str(im_num) + ".jpg"
        annotation_name = str(im_num) + ".xml"
        img_arr = elem[1][0]
        bboxes = elem[1][1]

        xml_annotation = annotation_for_image(img_name, img_arr.shape, bboxes)
        

        img_path = TRAIN_IMG_PATH
        annotations_path = TRAIN_ANNOTATIONS_PATH
        
        if im_num >= 240:
            img_path = VAL_IMG_PATH if im_num < 300 else TEST_IMG_PATH
            annotations_path = VAL_ANNOTATIONS_PATH if im_num < 300 else TEST_ANNOTATIONS_PATH

        # save image_arr as jpg
        scipy.misc.imsave(img_path+img_name, img_arr)

        # save annotation as xml
        xml_file = open(annotations_path+annotation_name, 'wb')
        xml_annotation.write(xml_file)

        



def annotation_for_image(img_name, im_size, bboxes):
    # Im_size: [width, height, depth] ??? should be squares anyways

    annotation = etree.Element('annotation')

    # Filename 
    filename = etree.Element('filename')
    filename.text = img_name

    # Image size
    size = etree.Element('size')
    # nested elements in size
    width = etree.Element('width')
    height = etree.Element('height')
    depth = etree.Element('depth')
    width.text = str (im_size[1])
    height.text = str (im_size[0])
    depth.text = str (im_size[2])

    size.append(width)
    size.append(height)
    size.append(depth)

    annotation.append(filename)
    annotation.append(size)

    for bbox in bboxes: 
        # object for each bounding box
        obj = etree.Element('object')

        # Always buildings we're detecting (name is label)
        name = etree.Element('name')
        name.text = "building"

        # Bounding box preocessing. We assume that bboxes are in
        # [centreX, centreY, width, height] format, and convert it to
        # x_min, x_max, y_min, y_max
        bndbox = etree.Element('bndbox')
        xmin = etree.Element('xmin')
        xmin.text = str (bbox[0] - (bbox[2]/2))

        ymin = etree.Element('ymin')
        ymin.text = str (bbox[1] - (bbox[3]/2))

        xmax = etree.Element('xmax')
        xmax.text = str (bbox[0] + (bbox[2]/2))

        ymax = etree.Element('ymax')
        ymax.text = str (bbox[1] + (bbox[3]/2))

        bndbox.append(xmin)
        bndbox.append(ymin)
        bndbox.append(xmax)
        bndbox.append(ymax)

        # append nested elements in obj
        obj.append(name)
        obj.append(bndbox)

        annotation.append(obj)

    # Returns a doc tree that can be written
    return etree.ElementTree(annotation)

if __name__ == "__main__":
    compile_dataset("./tiles.pkl")
