from lxml import etree


def bbox_for_image(file_name, im_size, bboxes):
		# Header.
		annotation = etree.Element('annotation')

		# Filename 
		filename = etree.Element('filename')
		filename.text = file_name

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
				xmin.text = str (bbox[0] - bbox[2]/2)

				ymin = etree.Element('ymin')
				ymin.text = str (bbox[1] - bbox[3]/2)

				xmax = etree.Element('xmax')
				xmax.text = str (bbox[0] + bbox[2]/2)

				ymax = etree.Element('ymax')
				ymax.text = str (bbox[1] + bbox[3]/2)

				bndbox.append(xmin)
				bndbox.append(ymin)
				bndbox.append(xmax)
				bndbox.append(ymax)

				# append nested elements in obj
				obj.append(name)
				obj.append(bndbox)

				annotation.append(obj)

		return annotation