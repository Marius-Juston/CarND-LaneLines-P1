import matplotlib.image as mpimg
import numpy as np


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=(255, 255, 255), thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


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
