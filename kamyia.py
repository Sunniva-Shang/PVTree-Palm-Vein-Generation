import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from args import parse_args
import random

from class_v_s import TPoint, Segment
from tools import end_distance, min_distance_point_to_line, show_img_3d, show_img_2d, draw_segment3d, draw_segment2d
import math


def get_random_position_c1(args):

    x = random.uniform(0, args.x_range)
    y = random.uniform(30, 30+args.y_range)
    z = random.uniform(0, args.z_range)
    p = np.array([x, y, z])

    return p
    
def get_random_position_c2(args):
   
    pp = random.uniform(0, 1)
    if pp <= 0.3:
        z = random.uniform(0, 30)
    elif pp <= 0.6:
        z = random.uniform(30, 70)
    else:
        z = random.uniform(70, 95)

    x = random.uniform(0, args.x_range)
    y = random.uniform(40, 40 + args.y_range)

    p = np.array([x, y, z])

    return p


def kamyia_optimal(args, p2, min_segment, segment):
    p0 = min_segment.point_in
    p1 = min_segment.point_out

    r0 = p0.radius
    r1 = p1.radius  
    r2 = p2.radius

    f0 = r0 * r0 * r0
    f1 = args.ratioQ * f0
    f2 = (1 - args.ratioQ) * f0

    x0 = p0.position[0]
    y0 = p0.position[1]
    z0 = p0.position[2]
    x1 = p1.position[0]
    y1 = p1.position[1]
    z1 = p1.position[2]
    x2 = p2.position[0]
    y2 = p2.position[1]
    z2 = p2.position[2]
  
    position_pb = np.array([(f0 * x0 + f1 * x1 + f2 * x2) / (2.0 * f0),
                            (f0 * y0 + f1 * y1 + f2 * y2) / (2.0 * f0),
                            (f0 * z0 + f1 * z1 + f2 * z2) / (2.0 * f0)])
    pb = TPoint(args, position_pb)

    
    l0 = end_distance(p0, pb)
    l1 = end_distance(p1, pb)
    l2 = end_distance(p2, pb)

   
    for i in range(50):
        R1 = r1 * r1
        R2 = r2 * r2
        gamma = args.gamma
        R0 = pow(f0 * (pow(R1, gamma) / f1 + pow(R2, gamma) / f2), 1.0 / gamma)

        position_pb[0] = (x0 * R0 / l0 + x1 * R1 / l1 + x2 * R2 / l2) / (R0 / l0 + R1 / l1 + R2 / l2)
        position_pb[1] = (y0 * R0 / l0 + y1 * R1 / l1 + y2 * R2 / l2) / (R0 / l0 + R1 / l1 + R2 / l2)
        position_pb[2] = (z0 * R0 / l0 + z1 * R1 / l1 + z2 * R2 / l2) / (R0 / l0 + R1 / l1 + R2 / l2)
        pb.position = position_pb

        r0 = math.sqrt(R0)
        r1 = math.sqrt(R1)
        r2 = math.sqrt(R2)
        f0 = r0 * r0 * r0
        f1 = r1 * r1 * r1
        f2 = r2 * r2 * r2
        l0 = end_distance(pb, p0)
        l1 = end_distance(pb, p1)
        l2 = end_distance(pb, p2)
        if l0 < 1.0 or l1 < 1.0 or l2 < 1.0:
            break

    p1.parent = pb
    p2.parent = pb
    pb.parent = p0

    if check_child(p1, segment):
        pb.num_end = p1.num_end + 1

    else:
        pb.num_end = 2

    pb.radius = args.r_ori + pb.num_end * args.ratioE
    seg0 = Segment(p0, pb)
    seg1 = Segment(pb, p1)
    seg2 = Segment(pb, p2)
    segment.remove(min_segment)
    segment.append(seg0)
    segment.append(seg1)
    segment.append(seg2)

    p = pb
    while(p.parent != None):
        p_parent = p.parent
        p_parent.num_end += 1
        p_parent.radius = args.r_ori + p_parent.num_end * args.ratioE
        p = p.parent


def check_child(p, segment):
    for s in segment:
        p_in = s.point_in
        if (p_in.position == p.position).all():
            return True

    return False

if __name__ == '__main__':
    args = parse_args()
    num = 100
    line = []
    fig = plt.figure(figsize=(14, 7))
    ax3d = fig.add_subplot(121, projection='3d')
    ax2d = fig.add_subplot(122)
    position_root = np.array([20.0, 40, 0.0])
    ax3d.scatter(position_root[0], position_root[1], position_root[2], color='blue')
    point_root = TPoint(args, position_root)
    position2 = np.array([25.0, 45, 10.0])
    ax3d.scatter(position2[0], position2[1], position2[2], color='blue')
    point2 = TPoint(args, position2)
    point2.parent = point_root
    seg = Segment(point_root, point2)
    line.append(seg)

    for i in range(num):
        position = get_random_position_c1(args)
        new_point = TPoint(args, position, radius=args.r_ori)
        min_seg = min_distance_point_to_line(position, line)
       
        kamyia_optimal(args, new_point, min_seg, line)

    print(len(line))
    draw_segment3d(args, ax3d, line)
    show_img_3d(ax3d)
    draw_segment2d(args, ax2d, line)
    show_img_2d(ax2d)
    plt.show()
