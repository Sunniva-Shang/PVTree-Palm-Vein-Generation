import os, shutil
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math
import multiprocessing


def get_ftype(ftype:str, ksize:int, sigma:float=0.01):
    
    if ftype == 'constant':
        value = np.ones(ksize)
    elif ftype == 'cosine':
        t = np.linspace(-np.pi/2, np.pi/2, ksize)
        value = np.cos(t)
    elif ftype == 'gaussian':
        t = np.linspace(-sigma*2, sigma*2, ksize)
        x = - t**2 / (2 * sigma**2)
        value = np.exp(x)
        
    mask_row = value[None, :]
    mask_col = value[:, None]

    return (mask_row, mask_col)


def single_filter(ksize:int, angle:float, width:int=1) -> np.ndarray:
    assert (ksize > 0)
    assert (0 <= angle <180)
    assert (width % 2 == 1)
    half = ksize // 2
    middle = half + 1 - 1
    filter = np.zeros(shape=(ksize,)*2, dtype=np.float32)
    half_width = (width - 1) // 2

    angle -= 180 if angle > 90 else 0
    radius = angle * math.pi / 180

    def in_box(x:int) -> bool:
        return 0 <= x < ksize

    if -45 < angle < 45:
        ratio = math.tan(radius)    
        for dx in range(-half, half+1):
            dy = round(dx * ratio)
            px = middle + dx
            py = middle + dy 
            filter[py, px] = 1.0
            if half_width > 0:
                for dw in range(1, half_width):
                    up = py+ dw 
                    down = py - dw 
                    if in_box(up):
                        filter[up, px] = 1.0
                    if in_box(down):
                        filter[down, px] = 1.0
    
    elif angle < -45 or angle > 45:
        ratio = math.cos(radius) / math.sin(radius)
        for dy in range(-half, half+1):
            dx = round(dy * ratio) 
            px = middle + dx
            py = middle + dy 
            filter[py, px] = 1.0
            if half_width > 0:
                for dw in range(1, half_width+1):
                    right = px + dw 
                    left = px - dw 
                    if in_box(right):
                        filter[py, right] = 1.0
                    if in_box(left):
                        filter[py, left] = 1.0

    elif angle == 45:
        for dx in range(-half, half+1):
            px = middle + dx
            py = middle + dx 
            filter[py, px] = 1.0
            if half_width > 0:
                for dw in range(1, half_width):
                    up = py+ dw 
                    down = py - dw 
                    if in_box(up):
                        filter[up, px] = 1.0
                    if in_box(down):
                        filter[down, px] = 1.0

    else:
        for dx in range(-half, half+1):
            px = middle + dx
            py = middle - dx 
            filter[py, px] = 1.0
            if half_width > 0:
                for dw in range(1, half_width):
                    up = py+ dw 
                    down = py - dw 
                    if in_box(up):
                        filter[up, px] = 1.0
                    if in_box(down):
                        filter[down, px] = 1.0 

    return filter


def get_filter(ksize:int, mode:int=0, ftype:str='constant', norm:bool=False, angle0:int=15, width:int=1) -> np.ndarray:
    assert (ksize > 0)
    assert (mode in [0, 1])
    assert (ftype in ['constant', 'gaussian', 'cosine'] )

    mask_row, mask_col = get_ftype(ftype=ftype, ksize=ksize)
    # anglelist = [i*angle0 for i in range(180//angle0)]
    # anglelist = [(180 + i*(-10)) for i in range(6, 0, -1)]
    # anglelist += [0]
    # anglelist += [i*10 for i in range(1, 7)]
    anglelist = [i*10 for i in range(0, 18, 2)]
    result = None
    for angle in anglelist:
        f = single_filter(ksize, angle, width)
        if (angle <= 45) or (angle >= 135):
            f = f * mask_row
        else:
            f = f * mask_col

        if result is None:
            result = f[None, :, :]
        else:
            result = np.concatenate((result, f[None, :, :]), axis=0)

    if mode == 1:
        # the following is to make half line
        filter = result
        num, height, width = filter.shape
        filter_result = np.zeros((num*2, height, width), dtype=filter.dtype)
        mask_up = np.zeros((height, width), dtype=filter.dtype)
        mask_down = np.zeros((height, width), dtype=filter.dtype)
        mask_left = np.zeros((height, width), dtype=filter.dtype)
        mask_right = np.zeros((height, width), dtype=filter.dtype)
        
        mask_up[(height+1)//2:, :] =  1.0
        mask_down[:(height+1)//2+1, :] =  1.0
        mask_left[:, (width+1)//2:] =  1.0
        mask_right[:, :(width+1)//2+1] =  1.0

        for i in range(filter.shape[0]):
            img = filter[i]

            if 45 <= anglelist[i] <= 135 :
                img1 = img * mask_down
                img2 = img * mask_up
                filter_result[i] = img1
                filter_result[i+num] = img2
            else:
                img1 = img * mask_right
                img2 = img * mask_left
                filter_result[i] = img1
                filter_result[i+num] = img2
        result = filter_result

    if norm:
            result /= np.sum(result.reshape(-1, ksize*ksize), axis=1)[:, None, None]

    return result


def apply_gaussian_low_pass_filter(image):
    
    gray = cv2.resize(image, (256, 256))
    
    # Apply Gaussian blur
    ksize=31
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0, borderType=cv2.BORDER_REFLECT)
    
    # Compute high-pass filtered image (details)
    details = np.float32(gray) - np.float32(blurred)

    return details


def select_response(filter_results:np.ndarray):

    # origin direction
    origin_direction = np.argmax(filter_results, axis=0)

    # split half part
    half_size = filter_results.shape[0] // 2
    left_results = filter_results[:half_size+1]
    right_results = filter_results[half_size:]

    min_value = -255.0
    
    diff_results = left_results[:-1] - right_results[1:][::-1]
    diff_results = np.sort(diff_results.flatten())
    idx = int(diff_results.shape[0] * 0.01)+1
    big_value, small_value = diff_results[-idx], diff_results[idx]
    left_diff = True if np.abs(big_value) > np.abs(small_value) else False

    if left_diff:
        filter_results[half_size+2:] = min_value
    else:
        filter_results[:half_size+1] = min_value


    response = np.max(filter_results, axis=0)
    direction = np.argmax(filter_results, axis=0)

    # get left or right mask
    mask = direction == origin_direction

    return response, direction, mask


def demo(filename):
    image = cv2.imread(filename, 0)
    image = apply_gaussian_low_pass_filter(image)

    mfrat_kernels = get_filter(ksize=31, ftype='gaussian', norm=True, angle0=30, width=1)
    mfrat_kernels *= -1

    # Apply the kernel to the image
    filter_results =[]
    for kernel in mfrat_kernels:
        filter_results.append(cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)[None])
    
    filter_results = np.concatenate(filter_results, axis=0)

    # select response
    response, direction, mask = select_response(filter_results)

    return filter_results, response, direction, mask


def show(images, response, direction):

    response = cv2.normalize(response, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    response = np.uint8(response * 255)
    print(f"response shape {response.shape}")

    direction = np.uint8(np.float32(direction) / 5.0 * 255)
    print(f"direction shape {direction.shape}")

    plt.imshow(images, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(response, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.imshow(direction, cmap='gray', vmin=0, vmax=255)
    plt.show()


def analyse_response(response, mask):

    response = response * np.float32(mask)

    hist, bins = np.histogram(response, bins=256)

    cumulative_distribution = np.cumsum(hist) / np.sum(hist)

    threshold = np.interp(0.90, cumulative_distribution, bins[:-1])

    mask = response < threshold

    return np.uint8(mask * 255)

def process(directory_path, output_path):
    # load all images under directory_path
    image_paths = os.listdir(directory_path)
    image_paths = [os.path.join(directory_path, image_path) for image_path in image_paths]

    for image_path in image_paths:
        filter_image, response, direction, mask = demo(image_path)
        # show(filter_image, response, direction)
        mask = np.ones_like(mask)
        result = analyse_response(response, mask)
        
        cv2.imwrite(os.path.join(output_path, os.path.basename(image_path)+'.png'), result)
        print(f"{os.path.basename(image_path)} processed")


def process_images(input_list, output_list):

    mfrat_kernels = get_filter(ksize=31, ftype='gaussian', norm=True, angle0=30, width=1)
    mfrat_kernels *= -1
    
    for sf, df in zip(input_list, output_list):
        image = cv2.imread(sf, 0)
        image = apply_gaussian_low_pass_filter(image)

        # Apply the kernel to the image
        filter_results =[]
        for kernel in mfrat_kernels:
            filter_results.append(cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)[None])
    
        filter_results = np.concatenate(filter_results, axis=0)
        response = np.max(filter_results, axis=0)

        hist, bins = np.histogram(response.flatten(), bins=256)

        cumulative_distribution = np.cumsum(hist) / np.sum(hist)

        threshold = np.interp(0.90, cumulative_distribution, bins[:-1])

        mask = response < threshold

        mask = np.uint8(mask * 255)
        cv2.imwrite(df, mask)


def read_images(spath, dpath):
    source_files = []
    dest_files = []
    os.makedirs(dpath, exist_ok=True)

    for root, dirs, files in os.walk(spath):
        
        for di in dirs:
            os.makedirs(os.path.join(dpath, os.path.relpath(root, spath), di), exist_ok=True)

        for file in files:
            source_files.append(os.path.join(root, file))
            dest_files.append(os.path.join(dpath, os.path.relpath(root, spath),file))
            if not os.path.exists(os.path.join(dpath, os.path.relpath(root, spath))):
                os.makedirs(os.path.join(dpath, os.path.relpath(root, spath)))

    return source_files, dest_files

def main():

    spath = "path to palmvein images"
    dpath = "path to pceline images"

    image_files, save_files = read_images(spath, dpath)

    num_process = 4
    process_list = []
    for i in range(num_process):
        process_list.append(multiprocessing.Process(target=process_images, args=(image_files[i::num_process], save_files[i::num_process])))

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()


if __name__ == '__main__':
    main()