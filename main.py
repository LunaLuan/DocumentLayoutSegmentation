import cv2
import os

import glob

import img_convert_binary as binary_convert
import matplotlib.pyplot as plt
from processing import xml_handle

from model.unet import unet_512_13 as model

INPUT_PATH = '1015_Private Test/'
OUTPUT_PATH = 'private_test/'

if __name__ == '__main__':
    files = [file for file in glob.glob(os.path.join(INPUT_PATH, '*.*'))]
    files.sort()

    for i in range(len(files)):
        file = files[i]
        print('Unet' + ' is running file: ' + file)


        im = cv2.imread(file)
        im_name = os.path.basename(file)
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        predict = model.run(im_rgb)

        result = xml_handle.get_xml_string(predict, im_name)

        output_file = open(OUTPUT_PATH + im_name[:-4] + '.xml', 'w')
        output_file.write(result)
        output_file.close()

        cv2.imwrite(OUTPUT_PATH + im_name, im)
        cv2.imwrite(OUTPUT_PATH + im_name[:-4] + '.tif', binary_convert.get_binary_image(im))

        predict = cv2.cvtColor(predict, cv2.COLOR_BGR2RGB)
        cv2.imwrite(OUTPUT_PATH + im_name, predict)

        print('Done: ' + file)


