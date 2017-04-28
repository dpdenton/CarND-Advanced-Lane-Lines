import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from os import listdir
from os.path import join

from utils import *

from calibration import CameraCalibration

# directory paths
CAMERA_CALIBRATION_DIR = './camera_cal'
TEST_IMAGES_DIR = './test_images'
EXAMPLES_DIR = './examples'
OUTPUT_IMAGES_DIR = './output_images'

camera_cal = CameraCalibration(nx=9, ny=6)

# Define a class to receive the characteristics of each line detection
class LineHistory():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

        self.coeffs = []

        self.radius = []

        self.num_points = []

        self.fail_count = 0


left_line_history = LineHistory()
right_line_history = LineHistory()


def process_img(img, fname='video.jpg', save=False):
    """
    
    :param img: ***BGR IMAGE***
    :param fname: 
    :return: 
    """

    save_dir = join(OUTPUT_IMAGES_DIR, os.path.basename(fname).split('.')[0])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save original image
    if save:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Images")
        plt.savefig(join(save_dir, "original.jpg"))

    # undistort image
    undistorted_img = camera_cal.undistort(img)
    if save:
        plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
        plt.title("Undistort Images")
        plt.savefig(join(save_dir, "undistort.jpg"))

    ########################
    ### Colour Channels  ###
    ########################

    # convert to HLS, extract L channel
    l_channel_img = bgr2x(undistorted_img, x="l")
    if save:
        plt.imshow(l_channel_img)
        plt.title("L channel")
        plt.savefig(join(save_dir, "l_channel.jpg"))

    # apply threshold
    l_channel_img_bin = apply_thresh(l_channel_img, thresh=(200, 255))
    if save:
        plt.imshow(l_channel_img_bin, cmap="gray")
        plt.title("L channel with thresh {}".format((200, 255)))
        plt.savefig(join(save_dir, "l_channel_img_bin.jpg"))

    s_channel_img = bgr2x(undistorted_img, x="s")
    if save:
        plt.imshow(s_channel_img)
        plt.title("S channel")
        plt.savefig(join(save_dir, "s_channel.jpg"))

    s_channel_img_bin = apply_thresh(s_channel_img, thresh=(150, 255))
    if save:
        plt.imshow(s_channel_img_bin, cmap="gray")
        plt.title("S channel with thresh {}".format((150, 255)))
        plt.savefig(join(save_dir, "s_channel_bin.jpg"))

    #####################
    ### Binary Images ###
    #####################

    # create binary gradient in x direction
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=5, thresh=(20, 100))
    if save:
        plt.imshow(gradx, cmap="gray")
        plt.title("Gradient X, thresh={}".format((20, 100)))
        plt.savefig(join(save_dir, "gradx_bin.jpg"))

    # create binary gradient in y direction
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=5, thresh=(20, 100))
    if save:
        plt.imshow(grady, cmap="gray")
        plt.title("Gradient Y, thresh={}".format((20, 100)))
        plt.savefig(join(save_dir, "grady_bin.jpg"))

    # create gradient direction binary
    dir_binary = dir_thresh(img, sobel_kernel=5, thresh=(0.7, 1.3))
    if save:
        plt.imshow(dir_binary, cmap="gray")
        plt.title("Gradient direction, thresh={}".format((0.7, 1.3)))
        plt.savefig(join(save_dir, "dir_bin.jpg"))

    # create gradient magnitude binary
    mag_binary = mag_thresh(img, sobel_kernel=5, thresh=(30, 100))
    if save:
        plt.imshow(mag_binary, cmap="gray")
        plt.title("Gradient magniture, thresh={}".format((30, 100)))
        plt.savefig(join(save_dir, "mag_bin.jpg"))

    # combine gradx/y or mag/dir binaries
    combined = np.zeros_like(mag_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    if save:
        plt.imshow(combined, cmap="gray")
        plt.title("Combined Grad, Mag, Dir {}".format(fname))
        plt.savefig(join(save_dir, "combined.jpg"))

    # combine grad, mag and dir with h and l channels
    combined_grad_mag_dir = np.zeros_like(combined)
    combined_grad_mag_dir[(combined == 1) | (s_channel_img_bin == 1) | (l_channel_img_bin == 1)] = 1
    if save:
        plt.imshow(combined_grad_mag_dir, cmap="gray")
        plt.title("Combined L OR S OR Grad, Mag, Dir")
        plt.savefig(join(save_dir, "combined_grad_mag_dir.jpg"))

    #############################
    ### Perspective Transform ###
    #############################

    combined = camera_cal.setM().warp(combined_grad_mag_dir)
    if save:
        plt.imshow(combined, cmap="gray")
        plt.title("combined Images")
        plt.savefig(join(save_dir, "transform.jpg"))

    ######################################################
    ### Implement Sliding Windows and Fit a Polynomial ###
    ######################################################

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((combined, combined, combined)) * 255
    window_img = np.zeros_like(out_img)

    # if we've already detected lane lines with confidence we can look ahead...
    # the line fails sanity checks 2 times we use the original method to rebuild the plots
    if left_line_history.coeffs and right_line_history.coeffs and left_line_history.fail_count < 2:

        margin = 50

        left_fit = left_line_history.coeffs[-1]
        right_fit = right_line_history.coeffs[-1]

        ploty = np.linspace(0, combined.shape[0] - 1, combined.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = combined.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # create window boundries
        left_window1 = left_fitx - margin
        left_window2 = left_fitx + margin
        right_window1 = right_fitx - margin
        right_window2 = right_fitx + margin

        leftx, lefty, rightx, righty = [], [], [], []

        # treat each row as a rectangle
        for row_idx in range(len(nonzeroy)):

            # get one nonzeroy value
            nzy = nonzeroy[row_idx]
            # get corresponding x value
            nzx = nonzerox[row_idx]

            # index by y value to get x edge of window at row on y level
            if left_window1[nzy] < nzx and left_window2[nzy] > nzx:
                leftx.append(nzx)
                lefty.append(nzy)

            elif right_window1[nzy] < nzx and right_window2[nzy] > nzx:
                rightx.append(nzx)
                righty.append(nzy)

        leftx = np.array(leftx)
        lefty = np.array(lefty)
        rightx = np.array(rightx)
        righty = np.array(righty)

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        #####################
        ### Visualisation ###
        #####################

        # Color in left and right line pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        if save:
            plt.imshow(result)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.savefig(join(save_dir, "using_history_plot.jpg"))

    else:

        # reset fail count
        left_line_history.fail_count = 0

        y_start = np.int(combined.shape[0] / 2)
        histogram = np.sum(combined[y_start:, :], axis=0)
        if save:
            plt.cla()
            plt.plot(histogram)
            plt.savefig(join(save_dir, "histogram.jpg"))
            plt.clf()

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(combined.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = combined.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = combined.shape[0] - (window + 1) * window_height
            win_y_high = combined.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ######################
        #### Visualization ###
        ######################

        # Generate x and y values for plotting
        ploty = np.linspace(0, combined.shape[0] - 1, combined.shape[0])
        temp_left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        temp_right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # visualise what it's produce at this stage
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        if save:
            plt.imshow(out_img)
            plt.plot(temp_left_fitx, ploty, color='yellow')
            plt.plot(temp_right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.savefig(join(save_dir, "plot.jpg"))
            plt.cla()

    #######################
    ### Radius of curve ###
    #######################

    L_radius = get_line_radius(leftx, lefty)
    R_radius = get_line_radius(rightx, righty)

    ######################
    ### Sanitiy Checks ###
    ######################

    append = True

    error = ((left_fit[0] - right_fit[0])**2)**0.5
    if error > 10 * abs(left_fit[0]) or error > 10 * abs(right_fit[0]):
    # if 1 - abs(left_fit[0] / right_fit[0]) > tolerance or 1 - abs(left_fit[1] / right_fit[1]) > tolerance:
        print("Lines are not parallel...")
        left_line_history.fail_count += 1
        append = False

    # add values to history
    if append:
        # coefficients
        left_line_history.coeffs.append(left_fit)
        right_line_history.coeffs.append(right_fit)

        # radii
        left_line_history.radius.append(L_radius)
        right_line_history.radius.append(R_radius)

    # new m is weighted average of previous n ms - this doesn't really make much of a difference
    # was trying to make the line a bit smoother but this doesn't work too well
    n = 10

    # get last n coefficients
    n_coeffs = np.array(left_line_history.coeffs[-n:])
    A_l = np.average(n_coeffs[:, 0], weights=range(1, len(n_coeffs) + 1) )
    B_l = np.average(n_coeffs[:, 1], weights=range(1, len(n_coeffs) + 1) )
    C_l = np.average(n_coeffs[:, 2], weights=range(1, len(n_coeffs) + 1) )

    n_coeffs = np.array(right_line_history.coeffs[-n:])
    A_r = np.average(n_coeffs[:, 0], weights=range(1, len(n_coeffs) + 1) )
    B_r = np.average(n_coeffs[:, 1], weights=range(1, len(n_coeffs) + 1) )
    C_r = np.average(n_coeffs[:, 2], weights=range(1, len(n_coeffs) + 1) )

    # Generate x and y values for plotting
    ploty = np.linspace(0, combined.shape[0] - 1, combined.shape[0])
    left_fitx = A_l * ploty ** 2 + B_l * ploty + C_l
    right_fitx = A_r * ploty ** 2 + B_r * ploty + C_r

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(combined).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, camera_cal.Minv, (combined.shape[1], combined.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    # get line positions...
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    l_position = A_l * img.shape[0] ** 2 + B_l * img.shape[0] + C_l
    r_position = A_r * img.shape[0] ** 2 + B_r * img.shape[0] + C_r
    middle_pxl = img.shape[1] / 2
    lane_width = (r_position - l_position) * xm_per_pix
    lane_center = (r_position + l_position) / 2
    car_position = (lane_center - middle_pxl) * xm_per_pix
    if car_position > 0:
        car_position_text = "Position: {0:.2f}m left of center".format(abs(car_position))
    else:
        car_position_text = "Position: {0:.2f}m right of center".format(abs(car_position))

    # add text to image
    result = cv2.putText(
        img=result,
        text="Lane Width: {0:.2f}m".format(lane_width),
        org=(50, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=.5,
        color=255,
    )

    # add text to image
    result = cv2.putText(
        img=result,
        text=car_position_text,
        org=(50, 90),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=.5,
        color=255,
    )

    result = cv2.putText(
        img=result,
        text="L Radius: {}m".format(round(L_radius)),
        org=(50, 130),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=.5,
        color=255,
    )

    result = cv2.putText(
        img=result,
        text="R Radius: {}m".format(round(R_radius)),
        org=(50, 170),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=.5,
        color=255,
    )

    if save:
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Result: {}".format(fname))
        plt.savefig(join(save_dir, "result.jpg"))

    plt.cla()

    return result


if __name__ == "__main__":

    # generate image points, calibrate and save matrix
    if camera_cal.load_calibration_mtx() is None:

        # get calibration files
        fnames = [join(CAMERA_CALIBRATION_DIR, f) for f in listdir(CAMERA_CALIBRATION_DIR)]

        # build list of points
        for fname in fnames:
            print("Generating image points from {}...".format(fname))
            img = cv2.imread(fname)
            camera_cal.gen_points(img)

        # use arbritrary image in list to calibrate and pickle matrix
        camera_cal.calibrate(img, save=True)

    ########################
    ### Test Calibration ###
    ########################

    # test undistortion
    # test_fname = join(CAMERA_CALIBRATION_DIR, 'calibration2.jpg')
    # camera_cal.test(test_fname, save=False)

    # test transform perspective - straight lines
    fname = "test_images/straight_lines1.jpg"
    img = cv2.imread(fname)
    undistorted_img = camera_cal.undistort(img)
    warped_img = camera_cal.setM().warp(undistorted_img)

    # test transform perspective - curved lines
    fname = "test_images/test2.jpg"
    img = cv2.imread(fname)
    undistorted_img = camera_cal.undistort(img)
    warped_img = camera_cal.setM().warp(undistorted_img)

    plt.cla()

    #######################
    ### Pipeline Starts ###
    #######################

    for fname in [join(TEST_IMAGES_DIR, f) for f in listdir(TEST_IMAGES_DIR)]:

        if 'test1' in fname:
            img = cv2.imread(fname)
            img = camera_cal.undistort(img)
            process_img(img, fname, save=True)

    # from moviepy.editor import VideoFileClip
    #
    # white_output = 'project_video_out.mp4'
    # clip1 = VideoFileClip("project_video.mp4")
    # white_clip = clip1.fl_image(process_img)
    # white_clip.write_videofile(white_output, audio=False)
    #
    # print("L: ", left_line_history)
    # print("R: ", right_line_history)


