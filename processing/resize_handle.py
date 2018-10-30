import numpy as np
import cv2

from util import helpers
from keras.utils import to_categorical

### Resize image by padding
def resize_image_by_pad(im, desired_size, color=(127, 127, 127)):
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    #     color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im


def resize_image_by_neg_pad(im, desired_size):
    old_size = im.shape[:2]

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    result = np.full((desired_size, desired_size, 3), -255)
    result[:im.shape[0], :im.shape[1], :im.shape[2]] = im

    return result


def resize_image_by_zeros_pad(im, desired_size):
    old_size = im.shape[:2]

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    result = np.full((desired_size, desired_size, 3), 0)
    result[:im.shape[0], :im.shape[1], :im.shape[2]] = im

    return result


def padding_for_X(images, desired_size):
    images = [resize_image_by_neg_pad(image, desired_size) for image in images]
    images = np.array(images)
    images = images.astype("float32")
    #     images = preprocess_input(images)

    return images


# def padding_for_Y(images, desired_size):
#     classes_map = []
#     for image in images:
#         image = resize_image_by_zeros_pad(image, desired_size)
#         one_hot = helpers.one_hot_it(image, label_values)
#         reverse = helpers.reverse_one_hot(one_hot)
#
#         classes_map.append(reverse)
#     classes_map = np.array(classes_map)
#     classes_map = to_categorical(classes_map, num_classes=n_classes)
#
#     return classes_map

def resize_image_from_neg_pad(im, old_size):
    im = im.astype('uint8')
    desired_size = im.shape[0]

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    temp = im[:new_size[0], :new_size[1], :im.shape[2]]
    result = cv2.resize(temp, (old_size[1], old_size[0]))

    return result
