from lxml import etree
import numpy as np
import scipy.misc
import pickle
from data_extract import extract_data


ANNOTATIONS_PATH = "./dataset/annotations/"
IMG_PATH = "./dataset/img/"

# Images and bboxes array -> write named images to folder, write xml annotations for each image to separate folder. 

def compile_dataset(path_to_data):
    data = extract_data(path_to_data)
    # data is an array of tuples. Each tuple contains an image arrays and a bbox array
    for elem in enumerate(data):
        # elem is (int index of tuple, tuple)
        img_name = str(elem[0]) + ".jpg"
        annotation_name = str(elem[0]) + ".xml"
        img_arr = elem[1][0]
        bboxes = elem[1][1]

        xml_annotation = annotation_for_image(img_name, img_arr.shape, bboxes)

        # save image_arr as jpg
        scipy.misc.imsave(IMG_PATH+img_name, img_arr)

        # save annotation as xml
        xml_file = open(ANNOTATIONS_PATH+annotation_name, 'wb')
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
