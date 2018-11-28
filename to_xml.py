from lxml import etree



## Create XML
annotation = etree.Element('annotation')
# Filename 
filename = etree.Element('filename')
filename.text = file ## VAR 

# Size
size = etree.Element('size')
# nested elements in size
width = etree.Element('width')
height = etree.Element('height')
depth = etree.Element('depth')
width.text = ""
height.text = ""
depth.text = ""
# append nested elements in size 
size.append(width)
size.append(height)
size.append(depth)

bndbox = 

# append filename
annotation.append(filename)
# append size
annotation.append(size)





<annotation>
	<filename>000001.jpg</filename>
	<size>
		<width>353</width>
		<height>500</height>
		<depth>3</depth>
	</size>
	<object>
		<name>dog</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>48</xmin>
			<ymin>240</ymin>
			<xmax>195</xmax>
			<ymax>371</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Left</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>8</xmin>
			<ymin>12</ymin>
			<xmax>352</xmax>
			<ymax>498</ymax>
		</bndbox>
	</object>
</annotation>