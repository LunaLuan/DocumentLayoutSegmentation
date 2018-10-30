from model.unet import unet_1024_11_batch1
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import os

import glob

import img_convert_binary as binary_convert
from processing import xml_handle

INPUT_PATH = '1015_Private Test/'
OUTPUT_PATH = 'private_test/'

if __name__ == '__main__':
    files = [file for file in glob.glob(os.path.join(INPUT_PATH, '*.*'))]
    files.sort()
#     predicts = unet_1024_11.run_on_files(files)
    for i in range(len(files)):
        file = files[i]
        print('Unet' + ' is running file: ' + file)

        im = cv2.imread(file)
        im_name = os.path.basename(file)
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        predict = unet_1024_11_batch1.run(im_rgb)
        print ('Saving result...')

        result = xml_handle.get_xml_string(predict, im_name)

        output_file = open(OUTPUT_PATH + im_name[:-4] + '.xml', 'w')
        output_file.write(result)
        output_file.close()

        cv2.imwrite(OUTPUT_PATH + im_name, im)
        cv2.imwrite(OUTPUT_PATH + im_name[:-4] + '.tif', 
                    binary_convert.get_binary_image(im))

        predict = cv2.cvtColor(predict, cv2.COLOR_BGR2RGB)
        cv2.imwrite(OUTPUT_PATH + im_name, predict)
    
    print ('Done...')

