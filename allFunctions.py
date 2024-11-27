import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

def correlation(im, kernel):
    im_height, im_width, im_channels = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size-1)/2)
    im_padded = np.zeros((im_height+pad_size*2, im_width+pad_size*2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y+kernel_size, x:x+kernel_size, c]
                new_value = np.sum(kernel*im_patch)
                im_out[y, x, c]= new_value
    return im_out

def convolution(im, kernel):
    im_height, im_width, im_channels = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size-1)/2)
    im_padded = np.zeros((im_height+pad_size*2, im_width+pad_size*2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    kernel_flipped = np.flip(np.flip(kernel, axis=0), axis=1)

    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y+kernel_size, x:x+kernel_size, c]
                new_value = np.sum(kernel_flipped * im_patch)
                im_out[y, x, c] = new_value
    return im_out

def gaussian_kernel(size, sigma=1):

    kernel = np.zeros((size, size)) 
    center = size // 2
    const = 1 / (2 * np.pi * sigma**2)
    
    total_sum = 0

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            
            value = const * np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel[i, j] = value
            
            total_sum += value
    
    kernel /= total_sum
    
    return kernel

def mean_filter_kernel(size):
    kernel = np.ones((size, size)) / (size * size)
    return kernel

def median_filter(image, kernel_size=3):

    height, width, channels = image.shape
    filtered_image = np.zeros_like(image)
    
    offset = kernel_size // 2
    
    for c in range(channels):
        for i in range(height):
            for j in range(width):
                row_start = max(0, i - offset)
                row_end = min(height, i + offset + 1)
                col_start = max(0, j - offset)
                col_end = min(width, j + offset + 1)
                
                neighborhood = image[row_start:row_end, col_start:col_end, c]
                filtered_image[i, j, c] = np.median(neighborhood)
    
    return filtered_image


def calculate_hs_histogram(img, bin_size):
    height, width, _ = img.shape
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    max_h = 179
    max_s = 255
    num_hue_bins = math.ceil((max_h + 1) / bin_size)
    num_saturation_bins = math.ceil((max_s + 1) / bin_size)
    hs_hist = np.zeros((num_hue_bins, num_saturation_bins))

    for i in range(height):
        for j in range(width):
            h = img_hsv[i, j, 0]
            s = img_hsv[i, j, 1]
            hs_hist[math.floor(h/bin_size), math.floor(s/bin_size)] += 1
    hs_hist /= hs_hist.sum()
    return hs_hist

def collect_hue_saturation(training_images):
    hs_values = []
    for img_path in training_images:
        img = cv2.imread(img_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s = img_hsv[:, :, 0].flatten(), img_hsv[:, :, 1].flatten()
        hs_values.extend(list(zip(h, s)))
    return np.array(hs_values)

def aggregate_histograms(images, bin_size):
    # Initialize empty histogram
    max_h = 179
    max_s = 255
    num_hue_bins = (max_h + 1) / bin_size
    num_saturation_bins = (max_s + 1) / bin_size
    num_hue_bins = math.ceil(num_hue_bins)
    num_saturation_bins = math.ceil(num_saturation_bins)
    hs_hist_total = np.zeros((num_hue_bins, num_saturation_bins))

    
    # Calculate the histogram for each image and sum it to hs_hist_total
    for img_path in images:
        img = cv2.imread(img_path)
        hs_hist = calculate_hs_histogram(img, bin_size)
        hs_hist_total += hs_hist
    
    # Normalize
    hs_hist_total /= hs_hist_total.sum()
    return hs_hist_total

def color_segmentation(img, hs_hist, bin_size, threshold):
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((height, width, 1))
    for i in range(height):
        for j in range(width):
            h = hsv[i, j, 0]
            s = hsv[i, j, 1]
            if hs_hist[math.floor(h/bin_size), math.floor(s/bin_size)] > threshold:
                mask[i, j, 0] = 1
    return mask

def harris_corner(img, block_size, k_size, k, threshold):
    # Perform corner estimation based on cv2 documentation

    dst = cv2.cornerHarris(img,block_size,k_size,k)
    dst = cv2.dilate(dst, None)
    peek_corners = np.copy(img)
    peek_corners = cv2.cvtColor(peek_corners, cv2.COLOR_GRAY2BGR)

    peek_corners[dst > threshold * dst.max()] = [0, 0, 255]
    return peek_corners


def compute_gradient(im):
    sobel_filter_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_filter_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gradient_x = convolution(im, sobel_filter_x)
    gradient_y = convolution(im, sobel_filter_y)

    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    magnitude *= 255.0 / magnitude.max()
    direction = np.arctan2(gradient_y, gradient_x)
    direction *= 180 / np.pi
    return magnitude, direction


def nms(magnitude, direction):
    height, width = magnitude.shape
    res = np.zeros(magnitude.shape)
    direction[direction < 0] += 180 

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current_direction = direction[y, x]
            current_magnitude = magnitude[y, x]
            if (0 <= current_direction < 22.5) or (157.5 <= current_direction <= 180):
                p = magnitude[y, x - 1]
                r = magnitude[y, x + 1]

            elif 22.5 <= current_direction < 67.5:
                p = magnitude[y + 1, x + 1]
                r = magnitude[y - 1, x - 1]

            elif 67.5 <= current_direction < 112.5:
                p = magnitude[y - 1, x]
                r = magnitude[y + 1, x]

            else:
                p = magnitude[y - 1, x + 1]
                r = magnitude[y + 1, x - 1]

            if current_magnitude >= p and current_magnitude >= r:
                res[y, x] = current_magnitude

    return res




def hysteresis(img, low_threshold, high_threshold):
    M, N = img.shape

    noise_map = np.zeros_like(img, dtype=np.uint8)
    weak_edge_map = np.zeros_like(img, dtype=np.uint8)
    strong_edge_map = np.zeros_like(img, dtype=np.uint8)
    final_map = np.copy(img)


    WEAK = 50
    STRONG = 255

    for i in range(M):
        for j in range(N):
            if img[i, j] >= high_threshold:
                strong_edge_map[i, j] = STRONG
                final_map[i, j] = STRONG 
            elif img[i, j] >= low_threshold:
                weak_edge_map[i, j] = WEAK
            else:
                noise_map[i, j] = 255 
                final_map[i, j] = 0 

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if final_map[i, j] == WEAK:
                if ((final_map[i + 1, j - 1] == STRONG) or (final_map[i + 1, j] == STRONG) or 
                    (final_map[i + 1, j + 1] == STRONG) or (final_map[i, j - 1] == STRONG) or 
                    (final_map[i, j + 1] == STRONG) or (final_map[i - 1, j - 1] == STRONG) or 
                    (final_map[i - 1, j] == STRONG) or (final_map[i - 1, j + 1] == STRONG)):
                    final_map[i, j] = STRONG
                else:
                    final_map[i, j] = 0

    return noise_map, weak_edge_map, strong_edge_map, final_map


def HoughTransform(edge_map):
    theta_values = np.deg2rad(np.arange(-90.0, 90.0))
    height, width = edge_map.shape
    diagonal_length = int(round(math.sqrt(width * width + height * height)))
    rho_values = np.linspace(-diagonal_length, diagonal_length, diagonal_length * 2 + 1)

    accumulator = np.zeros((len(rho_values), len(theta_values)), dtype=int)
    y_coordinates, x_coordinates = np.nonzero(edge_map)

    for edge_idx in range(len(x_coordinates)):
        x = x_coordinates[edge_idx]
        y = y_coordinates[edge_idx]
        for theta_idx in range(len(theta_values)):
            theta = theta_values[theta_idx]
            rho = int(round(x * np.cos(theta) + y * np.sin(theta)))
            accumulator[rho + diagonal_length, theta_idx] += 1

        # print("%d out of %d edges have voted" % (edge_idx+1, len(x_coordinates)))
        # cv2.imshow("Accumulator", (accumulator * 255 / accumulator.max()).astype(np.uint8))
        # cv2.waitKey(0)
    return accumulator, theta_values, rho_values


def non_maximum_suppression(accumulator, threshold=30, neighborhood_size=5):
    suppressed = np.copy(accumulator)
    for i in range(accumulator.shape[0] - neighborhood_size):
        for j in range(accumulator.shape[1] - neighborhood_size):
            local_patch = accumulator[i:i + neighborhood_size, j:j + neighborhood_size]
            max_value = np.max(local_patch)
            if max_value > threshold:
                suppressed[i:i + neighborhood_size, j:j + neighborhood_size] = 0
                suppressed[i + neighborhood_size // 2, j + neighborhood_size // 2] = max_value

    return suppressed


### Hough transform using cv2.HoughLines for P4
def apply_hough_transform(image_path):
    im = cv2.imread(image_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    edge_map = cv2.Canny(im_gray, 70, 150)

    lines = cv2.HoughLines(edge_map, 1, np.pi / 180, threshold=30)

    height, width = im_gray.shape

    if lines is not None:
        for line in lines:
            rho = line[0][0]
            theta = line[0][1]
            slope = -np.cos(theta) / np.sin(theta)
            intercept = rho / np.sin(theta)
            
            x1, x2 = 0, width
            y1 = int(slope * x1 + intercept)
            y2 = int(slope * x2 + intercept)

            cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow(f"Hough Lines - {image_path}", im)