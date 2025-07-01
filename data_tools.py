import os, shutil
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def cross(vein_path,palm_path, newpath):
    img1 = cv2.imread(vein_path)
    img2 = cv2.imread(palm_path)
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    h, w, c = img1.shape
    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img1,img1, mask = mask)
    img2_fg = cv2.bitwise_and(img2,img2, mask = mask_inv)
    dst = cv2.add(img1_bg, img2_fg)
    cv2.imwrite(newpath, dst)

def cross_folder(vein_path, palmprint_path, pv_path, sams=7, num_percase=10, num_case=4):
    # sams: num of samples
    # num: num_id/num_case,  4000/4=1000
    os.makedirs(pv_path, exist_ok=True)
    # list_v = os.listdir(vein_path)
    list_p = os.listdir(palmprint_path)
    for r in range(num_case):
        x = r * num_percase  # 0,1000,2000,3000
        for i in tqdm(range(x, x+num_percase)):
            palmimg = os.path.join(palmprint_path, list_p[i])
            for j in range(0, sams):
                veinimg = os.path.join(vein_path, 'c{}_{}'.format(r+1, i-x)+'_sample'+'{}.png'.format(j))
                newimg = os.path.join(pv_path, 'c{}_{}'.format(r+1, i-x)+'_sample'+'{}.png'.format(j))
                cross(veinimg, palmimg, newimg)


def aug(path, newpath):

    transfor = transforms.Compose([

        transforms.Resize((265,265), interpolation=Image.BILINEAR),
        transforms.RandomPerspective(distortion_scale=0.1, p=1, fill=255),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.RandomCrop((256,256), padding=0, pad_if_needed=False),
        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        # transforms.ColorJitter(brightness=(1, 1.3), contrast=0.1),
    ])

    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for i in tqdm(os.listdir(path)):
        imgpath = os.path.join(path, i)
        newimgpath = os.path.join(newpath, i)
        img = Image.open(imgpath)
        sam = transfor(img)
        sam.save(newimgpath)


def cvt(p):
    
    for img in os.listdir(p):
        imgpath = os.path.join(p, img)
        if os.path.isdir(imgpath):
            pass
        else:
            case, ids, _ = img.split('_')
            idname = case + '_' + ids
            idpath = os.path.join(p, idname)
            os.makedirs(idpath, exist_ok=True)
            newimgpath = os.path.join(idpath, img)
            shutil.move(imgpath, newimgpath)
