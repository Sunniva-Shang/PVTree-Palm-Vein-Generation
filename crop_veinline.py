import cv2
import os
import random

def crop(img_path='', case='', new_path=''):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (542, 616))
    (h, w) = img.shape[:2]
    center = (w//2, h//2)

    if case == 'c1':
        angler = -26
        start_x, start_y = 80, 100
        end_x, end_y = 450, 480

    elif case == 'c2':
        angler = -26
        start_x, start_y = 70, 145
        end_x, end_y = 445, 520
    elif case == 'c3':

        angler = -14
        start_x, start_y = 100, 115
        end_x, end_y = 445, 460

    elif case == 'c4':
        angler = -20
        start_x, start_y = 55, 135
        end_x, end_y = 440, 520

    M = cv2.getRotationMatrix2D(center, angler, 1)
    rotated_image = cv2.warpAffine(img, M, (w, h))

    crop_image = rotated_image[start_y:end_y, start_x:end_x]
    # print(crop_image.shape)
    resized_image = cv2.resize(crop_image, (400,400), interpolation=cv2.INTER_AREA)
    cv2.imwrite(new_path, resized_image)
   


def crop_img(path, newpath):
    for di in os.listdir(path):
        imgpath = os.path.join(path, di)
        c = di.split('_')[0]
        dpath = os.path.join(newpath, di)
        crop(imgpath, c, dpath)


