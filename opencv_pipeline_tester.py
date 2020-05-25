import matplotlib.image as mpimg
import numpy as np

from P1 import canny, gaussian_blur, region_of_interest, grayscale, weighted_img, hough_lines

kernel_size = 5
low = 50
high = 150
offset = 55
x_width = 150
y_height = 330

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 40  # minimum number of votes (intersections in Hough grid cell)
max_line_length = 35  # minimum number of pixels making up a line
max_line_gap = 30  # maximum gap in pixels between connectable line segments

import cv2

title = 'image'


def lane_finder(image, kernel_size, low, high, offset, x_width, ro, phi, threshold, max_line_length, max_line_gap):
    print(kernel_size, low, high, offset, x_width, ro, phi, threshold, max_line_length, max_line_gap)

    height, width, _ = image.shape

    vertices = np.array([[offset, height], [width / 2 - x_width / 2, y_height], [width / 2 + x_width / 2, y_height],
                         [width - offset, height]], dtype=np.int32)

    blurred_image = gaussian_blur(image, kernel_size)
    gray = grayscale(blurred_image)
    canny_points = canny(gray, low, high)

    copped_image = region_of_interest(canny_points, [vertices])

    hough, lines = hough_lines(copped_image, ro, phi, threshold, max_line_length, max_line_gap)

    color_edges = np.dstack((copped_image, copped_image, copped_image))
    output = weighted_img(color_edges, hough)
    output = cv2.polylines(output, [vertices], True, (0, 255, 255), 4)
    #
    slope = (lines[:, :, 1] - lines[:, :, 3]) / (lines[:, :, 0] - lines[:, :, 2])
    mask = np.logical_and(np.abs(slope) > .45, np.abs(slope) < .8)
    mask = np.logical_and(mask, np.isfinite(slope))
    mask = mask.flatten()

    slope = slope[mask]
    intercept = lines[:, :, 1] - (slope * lines[:, :, 0])
    left_lane = slope > 0
    average_slope_left = np.average(slope[left_lane])
    average_intercept_left = np.average(intercept[left_lane])
    average_slope_right = np.average(slope[np.logical_not(left_lane)])
    average_intercept_right = np.average(intercept[np.logical_not(left_lane)])

    output = cv2.line(output, (int(width / 2), int(average_slope_left * width / 2 + average_intercept_left)),
                      (int(width), int(average_slope_left * width + average_intercept_left)), (0, 255, 0), 3)

    output = cv2.line(output, (0, int(average_intercept_right)),
                      (int(width / 2), int(average_slope_right * width / 2 + average_intercept_right)), (0, 255, 0), 3)

    mask = output == [0, 0, 0]
    #
    output[mask] = image[mask]

    cv2.imshow('image2', output)


image = mpimg.imread('test_images/solidYellowCurve.jpg')


def call():
    lane_finder(image, kernel_size, low, high, offset, x_width, rho, theta, threshold, max_line_length, max_line_gap)


def new_kernel(val):
    global kernel_size
    kernel_size = val * 2 + 1

    call()


def new_low_threshold(val):
    global low
    low = val

    call()


def new_high_threshold(val):
    global high
    high = val

    call()


def new_offset(val):
    global offset
    offset = val

    call()


def new_x_width(val):
    global x_width
    x_width = val

    call()


def new_y_height(val):
    global y_height
    y_height = val

    call()


def new_theta(val):
    global theta
    theta = val

    call()


def new_threshold(val):
    global threshold
    threshold = val

    call()


def new_max_line_length(val):
    global max_line_length
    max_line_length = val

    call()


def new_max_line_gap(val):
    global max_line_gap
    max_line_gap = val

    call()


if __name__ == '__main__':
    cv2.namedWindow(title)

    cv2.createTrackbar("kernel", title, int((kernel_size - 1) / 2), 4, new_kernel)
    cv2.createTrackbar("low", title, low, 500, new_low_threshold)
    cv2.createTrackbar("high", title, high, 500, new_high_threshold)
    cv2.createTrackbar("offset", title, offset, 100, new_offset)
    cv2.createTrackbar("x_width", title, x_width, 100, new_x_width)
    cv2.createTrackbar("threshold", title, threshold, 200, new_threshold)
    cv2.createTrackbar("min_line_length", title, max_line_length, 200, new_max_line_length)
    cv2.createTrackbar("max_line_gap", title, max_line_gap, 200, new_max_line_gap)

    call()

    cv2.waitKey()
