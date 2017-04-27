import numpy as np
import cv2

def _num_of_channels(img):

    try:
        return img.shape[2]
    except IndexError:
        return 1

# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    if _num_of_channels(img) > 1:
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = img
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    axis = (orient == 'x', orient == 'y')
    sobelx = cv2.Sobel(grey_img, cv2.CV_64F, *axis, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # 5) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sxbinary


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image

    # 1) Convert to grayscale
    if _num_of_channels(img) > 1:
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = img

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(grey_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(grey_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sxbinary


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image


    # 1) Convert to grayscale
    if _num_of_channels(img) > 1:
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grey_img = img

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(grey_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    sobely = cv2.Sobel(grey_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)

    arctan = np.arctan2(abs_sobely, abs_sobelx)

    # print("ARCTAN: ", arctan)

    # 5) Create a binary mask where direction thresholds are met
    sxbinary = np.zeros_like(arctan)
    sxbinary[(arctan >= thresh[0]) & (arctan <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    return sxbinary

def bgr2x(img, x='s'):
    """
    Converts a bgr image to a single channel, 'x'
    :param img: image
    :param x: destination channel
    :return: single channel image
    """

    if x in ['h', 'l', 's']:
        # convert to HLS
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        if x == 'h':
            img = img[:, :, 0]
        elif x == 'l':
            img = img[:, :, 1]
        elif x == 's':
            img = img[:, :, 2]

    return img

def apply_thresh(img, thresh=(0, 255)):

    binary = np.zeros_like(img)
    binary[(img > thresh[0]) & (img <= thresh[1])] = 1

    return binary

def reduce(img, fn, kwargs, operator='or'):
    """
    
    :param img: binary image
    :param fn: function to apply to image
    :param kwargs: kwargs for function to apply
    :param operator: combination operator
    :return: binary image
    """

    assert operator == 'and' or operator == 'or'

    fn_binary = fn(*kwargs)

    combined = np.zeros_like(img)
    combined[(img == 1) & (fn_binary == 1)] = 1
    return combined

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_line_radius(x, y):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_world_space = y * ym_per_pix
    x_world_space = x * xm_per_pix

    # Fit new polynomials to x,y in world space
    fit = np.polyfit(y_world_space, x_world_space, 2)

    # first and second diferentials
    dx_dy = (2 * fit[0] * np.max(y_world_space)) + fit[1]
    dx2_dy2 = 2 * fit[0]

    return (1 + ((dx_dy) ** 2) ** 1.5) / np.absolute(dx2_dy2)


