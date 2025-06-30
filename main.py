import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from args import parse_args
import random
from tools import min_distance_point_to_line, \
    show_img_2d, draw_segment2d_rotate
from class_v_s import TPoint, Segment
from kamyia import get_random_position_c1, kamyia_optimal
from crop_veinline import crop
from data_tools import cross_folder, aug, cvt
import time
from PIL import Image
import os


def create_finger_position(case):
    if case == 1:
        finger_x = [0, 3, 7, 20, 25, 37, 41, 58, 65, 70]
        finger_z = [52, 68, 71, 76, 80, 80, 80, 76, 71, 50]
    elif case == 2:
        finger_x = [0, 8, 13, 23, 30, 40, 47, 60, 65, 70]
        finger_z = [58, 70, 72, 76, 80, 80, 80, 76, 72, 48]
    elif case == 3:
        finger_x = [0, 4, 9, 18, 26, 36, 43, 58, 65, 70, 53, 65]
        finger_z = [56, 64, 66, 75, 77, 80, 80, 80, 72, 48, 42, 34]
    elif case == 4:
        finger_x = [0, 6, 10, 25, 30, 40, 48, 60, 66, 70]
        finger_z = [56, 68, 72, 75, 80, 80, 80, 75, 70, 50]


    finger_position_list = []
    for i in range(len(finger_x)): 
        bias_x = random.uniform(-1.5, 1.5)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-2, 2)
        finger_position = np.array([finger_x[i] + bias_x, 40 + bias_y, finger_z[i] + bias_z])      
        finger_position_list.append(finger_position)

    return finger_position_list

def create_main_tree_c1(args, segment):
# c1
    list_root_point = []
    
    position_r_x = np.array([33.0, 22.0, 18.0, 25.0, 30.0, 38.0, 46.0, 50.0, 45.0, 38.0])
    position_r_z = np.array([-3.0, 30.0, 42.0, 50.0, 55.0, 58.0, -3.0, 31.0, 45.0, 58.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=9)
        elif i == 2:
            point_r = TPoint(args, position_r, num_end=8)
        elif i == 3:
            point_r = TPoint(args, position_r, num_end=6)
        elif i == 4 or i==8:
            point_r = TPoint(args, position_r, num_end=4)
        elif i == 5 or i==9:
            point_r = TPoint(args, position_r, num_end=2)
        elif i == 6 or i == 7:
            point_r = TPoint(args, position_r, num_end=5)
        
        list_root_point.append(point_r)
        

   
    for i in range(1, 6):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(7, len(list_root_point)):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

   
    s_finger_point = list_root_point[1: 2] + list_root_point[2: 6] + list_root_point[8:9] + list_root_point[7:8]

   
    list_branch_finger_point = []
    
    position_bf_x = np.array([5, 6, 15, 28, 47, 58, 70])
    position_bf_z = np.array([23, 55, 66, 73, 72, 53, 30])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        if i == 0 or i == 6:
            point_bf = TPoint(args, position_bf, num_end=1)
        else:
            point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)


    finger_position = create_finger_position(case=1)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[1], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[2], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[3], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        

    list_move = list_branch_finger_point

    for i in range(len(list_move)):
        list_move[i].parent = s_finger_point[i]
        seg = Segment(s_finger_point[i], list_move[i])
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE

def create_main_tree_c2(args, segment):
# c2
    list_root_point = []
    
    position_r_x = np.array([30.0, 25.0, 20.0, 22.0, 33.0, 41.0, 51.0, 50.0, 47.0])
    position_r_z = np.array([-5.0, 24.0, 35.0, 52.0, 58.0, -3.0, 27.0, 39.0, 58.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=7)
        elif i == 2:
            point_r = TPoint(args, position_r, num_end=6)
        elif i == 5 or i == 6:
            point_r = TPoint(args, position_r, num_end=5)    
        elif i == 3 or i == 7:
            point_r = TPoint(args, position_r, num_end=4)
        else:
            point_r = TPoint(args, position_r, num_end=2)
        
        list_root_point.append(point_r)
      
    for i in range(1, 5):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(6, len(list_root_point)):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    s_finger_point = list_root_point[1: 5] + list_root_point[8:9] + list_root_point[7:8] + list_root_point[6:7]

    list_branch_finger_point = []
    position_bf_x = np.array([5, 6, 18, 34, 48, 60, 70])
    position_bf_z = np.array([13, 56, 68, 70, 70, 45, 30])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        if i == 0 or i == 6:
            point_bf = TPoint(args, position_bf, num_end=1)
        else:
            point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)

    finger_position = create_finger_position(case=2)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[1], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[2], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[3], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        

    list_move = list_branch_finger_point

    for i in range(len(list_move)):
        list_move[i].parent = s_finger_point[i]
        seg = Segment(s_finger_point[i], list_move[i])
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE

def create_main_tree_c3(args, segment):
# c3
    list_root_point = []
    position_r_x = np.array([32.0, 20.0, 28.0, 36.0, 46.0, 60.0, 42.0, 45.0])
    position_r_z = np.array([-3.0, 40.0, 50.0, 57.0, 60.0, 65.0, -3.0, 25.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=10)
        elif i == 2:
            point_r = TPoint(args, position_r, num_end=8)
        elif i == 3:
            point_r = TPoint(args, position_r, num_end=6)    
        elif i == 4:
            point_r = TPoint(args, position_r, num_end=4)
        else:
            point_r = TPoint(args, position_r, num_end=2)
        
        list_root_point.append(point_r)
       
    for i in range(1, 6):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(7, len(list_root_point)):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    s_finger_point = list_root_point[1: 6] + list_root_point[7:8]

    list_branch_finger_point = []
    position_bf_x = np.array([5, 16, 33, 50])
    position_bf_z = np.array([57, 64, 72, 72])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)
    list_branch_finger_point.append(s_finger_point[-2])
    list_branch_finger_point.append(s_finger_point[-1])
    finger_position = create_finger_position(case=3)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[0]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[0], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[1], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[2], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[3], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 10 or i == 11:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        

    list_move = list_branch_finger_point[0:4]

    for i in range(len(list_move)):
        list_move[i].parent = s_finger_point[i]
        seg = Segment(s_finger_point[i], list_move[i])
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE

def create_main_tree_c4(args, segment):
# c4
    list_root_point = []
    position_r_x = np.array([30.0, 22.0, 42.0, 45.0, 40.0, 30.0, 42.0])
    position_r_z = np.array([-3.0, 30.0, -3.0, 25.0, 45.0, 50.0, 54.0])
    for i in range(len(position_r_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_r = np.array([position_r_x[i] + bias_x, 40 + bias_y, position_r_z[i] + bias_z])
        if i == 0 or i == 1:
            point_r = TPoint(args, position_r, num_end=1)
        elif i == 2 or i ==3:
            point_r = TPoint(args, position_r, num_end=10)
        elif i == 4:
            point_r = TPoint(args, position_r, num_end=8)    
        else:
            point_r = TPoint(args, position_r, num_end=4)
        
        list_root_point.append(point_r)
       
    for i in range(1, 2):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)

    for i in range(3, len(list_root_point)-1):
        list_root_point[i].parent = list_root_point[i-1]
        seg = Segment(list_root_point[i-1], list_root_point[i])
        segment.append(seg)
    list_root_point[-1].parent = list_root_point[-3]
    segg = Segment(list_root_point[-3], list_root_point[-1])
    segment.append(segg)

    s_finger_point = list_root_point[1: 2] + list_root_point[5:7] + list_root_point[3: 4]

    list_branch_finger_point = []
    position_bf_x = np.array([0, 19, 20, 36, 50, 57])
    position_bf_z = np.array([38, 60, 70, 74, 72, 53])
    for i in range(len(position_bf_x)):
        bias_x = random.uniform(-3, 3)
        bias_y = random.uniform(-7, 7)
        bias_z = random.uniform(-5, 5)
        position_bf = np.array([position_bf_x[i] + bias_x, 40 + bias_y, position_bf_z[i] + bias_z])
        if i == 0:
            point_bf = TPoint(args, position_bf)
        else:
            point_bf = TPoint(args, position_bf, num_end=2)
        list_branch_finger_point.append(point_bf)
   
    finger_position = create_finger_position(case=4)

    list_move = []
    list_finger_point = []
    for i in range(len(finger_position)):
        finger_point = TPoint(args, finger_position[i])
        if i == 0 or i == 1:
            finger_point.parent = list_branch_finger_point[1]
            list_finger_point.append(finger_point)
            seg1 = Segment(list_branch_finger_point[1], finger_point)
            seg1.index = len(segment)
            segment.append(seg1)
        elif i == 2 or i == 3:
            finger_point.parent = list_branch_finger_point[2]
            list_finger_point.append(finger_point)
            seg2 = Segment(list_branch_finger_point[2], finger_point)
            seg2.index = len(segment)
            segment.append(seg2)

        elif i == 4 or i == 5:
            finger_point.parent = list_branch_finger_point[3]
            list_finger_point.append(finger_point)
            seg3 = Segment(list_branch_finger_point[3], finger_point)
            seg3.index = len(segment)
            segment.append(seg3)

        elif i == 6 or i == 7:
            finger_point.parent = list_branch_finger_point[4]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[4], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        elif i == 8 or i == 9:
            finger_point.parent = list_branch_finger_point[5]
            list_finger_point.append(finger_point)
            seg4 = Segment(list_branch_finger_point[5], finger_point)
            seg4.index = len(segment)
            segment.append(seg4)
        

    list_move = list_branch_finger_point

    for i in range(len(list_move)):
        if i == 0:
            list_move[i].parent = s_finger_point[0]
            seg = Segment(s_finger_point[i], list_move[i])
        elif i == 1 or i == 2:
            list_move[i].parent = s_finger_point[1]
            seg = Segment(s_finger_point[1], list_move[i])
        elif i == 3 or i == 4:
            list_move[i].parent = s_finger_point[2]
            seg = Segment(s_finger_point[2], list_move[i])
        elif i == 5:
            list_move[i].parent = s_finger_point[3]
            seg = Segment(s_finger_point[3], list_move[i])
        
        segment.append(seg)

    for i in range(len(segment)):
        segment[i].point_out.radius = args.r_ori + segment[i].point_out.num_end * args.ratioE


def save_subfig(fig, ax, save_path):
    ax.axis('tight')
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path, bbox_inches=bbox, pad_inches=0)

def flip_image(image_path, output_path):
    
    img = Image.open(image_path)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(output_path)

def create_3dtree(case=1, fullpath='', croppath='', s=1, num_sams=7):
    args = parse_args()
    segment = []
    randnum = random.randint(-10, 10)
    num = 80
    # num = 100 + randnum
    # num = 70 + randnum
    
    if case == 1:
        create_main_tree_c1(args, segment)
    elif case == 2:
        create_main_tree_c2(args, segment)
    elif case == 3:
        create_main_tree_c3(args, segment)
    elif case == 4:
        create_main_tree_c4(args, segment)

    for i in range(num):
        position = get_random_position_c1(args)
        new_point = TPoint(args, position, args.r_ori)
        min_seg = min_distance_point_to_line(position, segment)
        kamyia_optimal(args, new_point, min_seg, segment)

    # sams
    for sam in range(num_sams):
        fig = plt.figure(figsize=(7, 8))
        ax = fig.add_subplot(111)
        ang = np.radians(random.randint(-2, 2))
        draw_segment2d_rotate(args, ax, segment, angle=ang, s=s) 
        show_img_2d(ax)
        fig.savefig(fullpath+'_sample{}.png'.format(sam), bbox_inches='tight', pad_inches=0)
        plt.close()
        crop(fullpath+'_sample{}.png'.format(sam), 'c{}'.format(case), croppath+'_sample{}.png'.format(sam))
    
    time.sleep(1)


def get_large_pv(fp, cp, num_id, sams):
    num = num_id // 4
    os.makedirs(fp, exist_ok=True)
    os.makedirs(cp, exist_ok=True)
    for i in tqdm(range(num)):
        fp1 = os.path.join(fp, 'c1_' + str(i))
        fp2 = os.path.join(fp, 'c2_' + str(i))
        fp3 = os.path.join(fp, 'c3_' + str(i))
        fp4 = os.path.join(fp, 'c4_' + str(i))
        cp1 = os.path.join(cp, 'c1_' + str(i))
        cp2 = os.path.join(cp, 'c2_' + str(i))
        cp3 = os.path.join(cp, 'c3_' + str(i))
        cp4 = os.path.join(cp, 'c4_' + str(i))
        create_3dtree(case=1, fullpath=fp1, croppath=cp1, num_sams=sams)
        create_3dtree(case=2, fullpath=fp2, croppath=cp2, num_sams=sams)
        create_3dtree(case=3, fullpath=fp3, croppath=cp3, num_sams=sams)
        create_3dtree(case=4, fullpath=fp4, croppath=cp4, num_sams=sams)
    

if __name__ == '__main__':
    
    num_case = 4
    num_percase = 5
    num_ids = num_case * num_percase # num of ids
    num_sams = 7 # num of samples

    # path to full palm vein patterns
    full_path = './pv_pattern_results/full'
    # path to crop palm vein patterns
    crop_path = './pv_pattern_results/crop'
    # path to images of palmprint and palm vein blended together
    pv_path = './pv_pattern_results/pv'
    # path to enhanced images
    aug_path = './pv_pattern_results/palmvein'

    # generate 3d tree
    get_large_pv(full_path, crop_path, num_ids, num_sams)
    
    # blend palm vein patterns with palmprint modeled by bezier creases
    cross_folder(vein_path=crop_path, palmprint_path='./bezierpalm', 
                  pv_path=pv_path, sams=num_sams, num_percase=num_percase, 
                  num_case=num_case)
    
    # image enhancement
    aug(pv_path, aug_path)

    cvt(aug_path)
   
    
