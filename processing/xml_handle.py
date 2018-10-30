import xml.etree.ElementTree as ET
from xml.dom import minidom
from processing.contour_handle import *
from util.meta_data import *
import datetime


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def get_xml_string(image, image_name):
    PcGts = ET.Element('PcGts')
    PcGts.set("xmlns", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15")
    PcGts.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    PcGts.set("xsi:schemaLocation", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15/pagecontent.xsd")

    Metadata = ET.SubElement(PcGts, 'Metadata')

    Creator = ET.SubElement(Metadata, 'Creator')
    Creator.text = 'LCD'
    Created = ET.SubElement(Metadata, 'Created')
    Created.text = datetime.date.today().strftime('%Y-%m-%dT%H:%M:%S')
    LastChange = ET.SubElement(Metadata, 'LastChange')
    LastChange.text = datetime.date.today().strftime('%Y-%m-%dT%H:%M:%S')

    Page = ET.SubElement(PcGts, 'Page')
    Page.set('imageFilename', image_name)
    Page.set('imageWidth', str(image.shape[1]))
    Page.set('imageHeight', str(image.shape[0]))
    # Page.text = ' '

    id = 0
    for region in REGION_COLOR.keys():
        contours = get_contours_by_color(image, REGION_COLOR[region])
        region = region.split()
        for contour in contours:
            if is_contour_bad(contour):
                continue

            region_item = ET.SubElement(Page, region[0])
            region_item.set('id', 'r' + str(id))
            id += 1

            if len(region) == 2:
                region_item.set('type', region[1])

            coords = ET.SubElement(region_item, 'Coords')
            points = ""
            for point in contour:
                point = point[0]

                p1 = point[0]
                p2 = point[1]

                points += str(p1) + ',' + str(p2) + ' '
            points = points[:(len(points) - 1)]

            coords.set('points', points)

    return prettify(PcGts)

