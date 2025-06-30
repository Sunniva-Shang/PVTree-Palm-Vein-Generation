import random
from matplotlib import pyplot as plt
import numpy as np
from class_v_s import TPoint, Segment
import copy

def norm(a, b, c):
    n = np.sqrt(a * a + b * b + c * c)
    if n == 0:
        return np.array([0, 0, 0])
    else:
        return np.array([a / n, b / n, c / n])

def min_distance_point_to_line(p, line):
    if len(line) == 0:
        return p
    elif len(line) == 1:
        return line[0]
    else:
        min_len = 100000
        min_seg = line[0]
        for s in line:
            x1 = s.point_in.position
            x2 = s.point_out.position
            a = p - x1
            b = x2 - x1
            d = p - x2
            a_len = np.linalg.norm(a)
            d_len = np.linalg.norm(d)
            b_len = np.linalg.norm(b)

            if np.dot(a, b) < 0 or np.dot(a, -b) < 0:
                l = min(a_len, d_len)
                if l <= 3.0 or l >= min_len:
                    continue
                else:
                    min_len = l
                    min_seg = s

            elif np.dot(a, b) == 0:
                continue

            else:
                c_len = np.dot(a, b) / np.linalg.norm(b)
                b_direction = b / b_len
                c = c_len * b_direction
                e = c - a
                l = np.linalg.norm(e)
                if l < min_len:
                    min_len = l
                    min_seg = s

        return min_seg

def min_distance_point_to_point(p, p_list):

    if len(p_list) == 1:
        return p
    else:
        min_d = 100000
        for v in p_list:
            if v == p:
                pass
            else:
                d = end_distance(p, v)
                if d < min_d:
                    min_d = d
                    min_p = v
        return min_p

def end_distance(p, p_end):
    x = p.position[0]
    y = p.position[1]
    z = p.position[2]
    x_end = p_end.position[0]
    y_end = p_end.position[1]
    z_end = p_end.position[2]
    d = np.sqrt(np.power(x - x_end, 2) + np.power(y - y_end, 2) + np.power(z - z_end, 2))
    return d

def find_perpendicular_circle(center, normal, radius):

    normal = normal / np.linalg.norm(normal)

    num_points = 100  
    theta = np.linspace(0, 2 * np.pi, num_points)
    v = np.cross(normal, np.array([0, 0, 1])) 
    v /= np.linalg.norm(v)
    u = np.cross(normal, v)
    u /= np.linalg.norm(u)
    y = u[:, np.newaxis] * np.cos(theta) + v[:, np.newaxis] * np.sin(theta)
    y2 = radius * y
    a = center
    for i in range(num_points - 1):
        a = np.vstack((a, center))
    circle_points = a.T + y2

    return circle_points

def draw_cylender(ax,p1,p2,r, color='red'):

    v = p2 - p1
    r = r * 0.3
   
    circle_coords1 = find_perpendicular_circle(p1, v, r)
    circle_coords2 = find_perpendicular_circle(p2, v, r)
   
    x = np.ones(shape=(100, 2))
    y = np.ones(shape=(100, 2))
    z = np.ones(shape=(100, 2))
    for i in range(100):
        x[i][0] = circle_coords1.T[i][0]
        x[i][1] = circle_coords2.T[i][0]
        y[i][0] = circle_coords1.T[i][1]
        y[i][1] = circle_coords2.T[i][1]
        z[i][0] = circle_coords1.T[i][2]
        z[i][1] = circle_coords2.T[i][2]
   
    ax.plot_surface(x, y, z, color=color, alpha=1, antialiased=False)

def d_update(p, p_list):
    d_rand = np.random.randn(3)
   
    w_rand = 0.3
    if len(p_list) == 1:
        return p.d + w_rand * d_rand
    else:
        p_min = min_distance_point_to_point(p, p_list)
        d_nebor = norm(p_min.position[0] - p.position[0],
                       p_min.position[1] - p.position[1],
                       p_min.position[2] - p.position[2])

        d = p.d + d_nebor + w_rand * d_rand
        return d

def create_finger_position():
    finger_range = [0, 25, 50, 75, 100]
    finger_position = []
    for i in range(len(finger_range) - 1):
        x1 = finger_range[i]
        x2 = finger_range[i + 1]
        y1 = random.uniform(-5, 5)
        y2 = random.uniform(-5, 5)
        finger_position1 = np.array([x1 + 5, 50 + y1, 100])
        finger_position2 = np.array([x2 - 5, 50 + y2, 100])
        finger_position.append(finger_position1)
        finger_position.append(finger_position2)
    return finger_position

def exist(p, b):
    for i in range(len(b)):
        if ((p.position[0] == b[i].position[0]) and
                (p.position[1] == b[i].position[1]) and
                (p.position[2] == b[i].position[2])):
            return True
    return False

def move(args, p, p_end, p_list, vertex, line):
    if end_distance(p, p_end) <= 3.0:
        for i in range(len(vertex)):
            if vertex[i] == p:
                vertex[i].parent = p_end

        segment_new = Segment(p_end, p)
        line.append(segment_new)
        p_list.remove(p)

    else:
        x = p.position[0] + p.l * d_update(p, p_list)[0]
        y = p.position[1] + p.l * d_update(p, p_list)[1]
        z = p.position[2] + p.l * d_update(p, p_list)[2]
        position_move = np.array([x, y, z])
        p_move = TPoint(args, position_move, p.radius, p_end)
        vertex.append(p_move)

        for i in range(len(p_list)):
            if p_list[i] == p:
                p_list[i] = p_move

        for j in range(len(vertex)):
            if vertex[j] == p:
                vertex[j].parent = p_move

        for k in range(len(line)):
            if line[k].point_in == p:
                line[k].point_in.parent = p_move

        p.parent = p_move
        segment_new = Segment(p_move, p)
        line.append(segment_new)


def gen_tree(args, p_list, p_end, vertex, segment):

    while len(p_list) > 0:
        p_list2 = p_list.copy()
        for p in p_list2:
            if exist(p, p_list):
                if len(p_list) == 2:
                    if end_distance(p_list[0], p_list[1]) <= 5.0:
                        p_list[0].parent = p_list[1]
                        new_segment = Segment(p_list[1], p_list[0])
                        segment.append(new_segment)

                        for i in range(len(vertex)):
                            if vertex[i] == p_list[0]:
                                vertex[i].parent = p_list[1]

                        p_list.remove(p_list[0])

                move(args, p, p_end, p_list, vertex, segment)

def draw_cylender3d(args, ax, p1, p2):
    while (end_distance(p1, p2) > args.step):
        d_rand = np.random.randn(3)
        d_rand /= np.linalg.norm(d_rand)
        w_rand = 0.5
        p2.d = norm(p1.position[0] - p2.position[0], p1.position[1] - p2.position[1],
                    p1.position[2] - p2.position[2])
        p2.d = p2.d + w_rand * d_rand

        x = p2.position[0] + p2.l * p2.d[0]
        y = p2.position[1] + p2.l * p2.d[1]
        z = p2.position[2] + p2.l * p2.d[2]
        position_move = np.array([x, y, z])
        p_move = TPoint(args, position_move, p2.radius, p1)
        draw_cylender(ax, p2.position, position_move, p2.radius)
        p2 = p_move

    draw_cylender(ax, p1.position, p2.position, p2.radius)


def draw_segment3d(args, ax, line):
    for i in range(len(line)):
        segment = line[i]
        point_in = segment.point_in
        point_out = segment.point_out
        draw_cylender3d(args, ax, point_in, point_out)


def draw_cylender2d(args, ax, p1, p2, ymin=30, ymax=50, s=1):
    # p1: point_in
    # p2: point_out
    np.random.seed(s)
    l = args.step2d
    while (end_distance(p1, p2) > l):

        d_rand = np.random.randn(3)
        d_rand /= np.linalg.norm(d_rand)
        w_rand = 0.5
        p2.d = norm(p1.position[0] - p2.position[0], p1.position[1] - p2.position[1],
                    p1.position[2] - p2.position[2])
        p2.d = p2.d + w_rand * d_rand

        x = p2.position[0] + l * p2.d[0]
        y = p2.position[1] + l * p2.d[1]
        z = p2.position[2] + l * p2.d[2]
        position_move = np.array([x, y, z])
        p_move = TPoint(args, position_move, radius=p2.radius, end_point=p1)

        xx = [p2.position[0], x]
        zz = [p2.position[2], z]
        color = caculate_color(y, ymin=ymin, ymax=ymax)
        ax.plot(xx, zz, color=(color, color, color), linewidth=p2.radius * 1.5)

        p2 = p_move

    xx = [p2.position[0], p1.position[0]]
    zz = [p2.position[2], p1.position[2]]
    y_p1 = p1.position[1]
    color = caculate_color(y_p1, ymin=ymin, ymax=ymax)
    ax.plot(xx, zz, color=(color, color, color), linewidth=p2.radius * 1.6)


def rotate_point_around_z_axis(point, angle=np.pi/30):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle),  np.cos(angle), 0],
                                [0,               0,              1]])
    
    return np.dot(rotation_matrix, point)

def rotation_matrix_x(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, cos_angle, -sin_angle],
                     [0, sin_angle, cos_angle]])

def rotation_matrix_y(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, 0, sin_angle],
                     [0, 1, 0],
                     [-sin_angle, 0, cos_angle]])

def rotation_matrix_z(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, -sin_angle, 0],
                     [sin_angle, cos_angle, 0],
                     [0, 0, 1]])

def rotate_point_around_xz(point, ang):

    angle_x = ang[0]
    angle_z = ang[1]
    rotation_matrix = rotation_matrix_x(angle_x).dot(rotation_matrix_z(angle_z))

    return point.dot(rotation_matrix)

def movetozreo(line3, tv = (-35, -40, -40)):
    for i in range(len(line3)):
        segment = line3[i]
        point_in = segment.point_in
        point_out = segment.point_out
        if point_in.move == None:
            point_in.position += tv
            point_in.move = True
        if point_out.move == None:
            point_out.position += tv
            point_out.move = True


def rotate_3dtree(line, angle):
    line2 = copy.deepcopy(line)
    movetozreo(line2)
    for i in range(len(line2)):
        segment = line2[i]
        point_in = segment.point_in
        point_out = segment.point_out
        if point_in.rotate == None:
           
            point_in.position = rotate_point_around_z_axis(point_in.position, angle=angle)
            point_in.rotate = True
        if point_out.rotate == None:
           
            point_out.position = rotate_point_around_z_axis(point_out.position, angle=angle)
            point_out.rotate = True
    
    ylist = []
    for i in line2:
        x = i.point_out.position[1]
        ylist.append(x)
    y_min = min(ylist)
    y_max = max(ylist)    
    return line2, y_min, y_max 

def draw_segment2d_rotate(args, ax, line, angle, s):
    line2, y_min, y_max = rotate_3dtree(line, angle)
    np.random.seed(s)
    for i in range(len(line2)):
        segment = line2[i]
        point_in = segment.point_in
        point_out = segment.point_out
        draw_cylender2d(args, ax, point_in, point_out, ymin=y_min, ymax=y_max, s=s)


def draw_segment2d(args, ax, line):
    for i in range(len(line)):
        segment = line[i]
        point_in = segment.point_in
        point_out = segment.point_out
        draw_cylender2d(args, ax, point_in, point_out)


def caculate_color(y, ymin=33, ymax=47):
    
    x = random.uniform(0.95, 1.05)
    ymin = ymin + 40
    ymax = ymax + 40
    y = y + 40
    yy = (y-ymin) * x/(ymax-ymin)

        
    if yy > 0.5:
        yy = yy * 1.1
        yy = round(255 * yy) / 255
        if yy >= 1:
            yy = 1
    else:
        yy = yy * 0.9
        yy = round(255 * yy) / 255
        if yy <= 0:
            yy = 0
       
    return yy


def draw_segment(ax, line):
    for i in range(len(line)):
        segment = line[i]
        point_in = segment.point_in
        point_out = segment.point_out
        draw_cylender(ax, point_in.position, point_out.position, point_out.radius)


def show_img_3d(ax):
   
    ax.set_xlim(0, 70)
    ax.set_ylim(30, 50)
    ax.set_zlim(0, 80)
    ax.set_xticks(np.arange(0, 80, 10))
    ax.set_yticks(np.arange(30, 60, 10))
    ax.set_zticks(np.arange(0, 90, 10))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.axis('equal')
    plt.grid(True)
    

def show_img_2d(ax): 
    plt.margins(0, 0)
    ax.set_xlim(0, 70)   
    ax.set_ylim(0, 80)   
    ax.set_axis_off()  
    plt.grid(False)
    plt.axis('equal')
   