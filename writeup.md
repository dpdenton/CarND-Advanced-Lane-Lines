## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained the calibrate.py file.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/undistortion_exmpale_chessboard.jpg)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/undistortion_exmpale.jpg)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 85 through 160 in `pipeline.py`).

Here's an example of a **gradient direction** binary image using a radian threshold of (0.7, 1.3):

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/test2/dir_bin.jpg)

Here's an example of a **gradient magnitude** binary image using a threshold of (30, 100):

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/test2/mag_bin.jpg)

Here's an example of a **gradient in the x direction** binary image using a magnitude threshold of (20, 100):

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/test2/gradx_bin.jpg)

Here's an example of a **gradient in the y direction** binary image using a magnitude threshold of (20, 100):

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/test2/grady_bin.jpg)

I combined the gradx and grady binaries to reduce the noise, and overlaid that onto a combination of the magnitude and direction binaries. Here is an example:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/test2/combined.jpg)

##### Colour Transforms

The S channel struggles to pick up on weaker white lines, however the L channels picks these up quite well. Below is an example of an S channel binary with a threshold of (150, 255), capturing the yellow line nicely, but not the white line:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/test2/s_channel_bin.jpg)

Below is an example of an L channel binary with a threshold of (200, 255), capturing the white line on the right:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/test2/l_channel_img_bin.jpg)

Comibinig a binary S channel and L channel image with an OR operator helps capture lines of both colour well.

Using this approach you can set quite a high threshold when generating the binary images to reduce the noise and get a cleaner image to detect lines.

These images were overlaid with the previous binary combined image to create the below:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/test2/combined_grad_mag_dir.jpg)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the calibrate.py file and is used on line 168 in pipeline.py. (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warp()` function takes as inputs an image (`img`).

The matrix is set by calling setM(), a method in the CalibrateCamera class. If no src or dst args are passed, the cv2.getPerspectiveTransform() is called using hard-coded points from "test_images/straight_lines1.jpg":

```python
# hard-coded src points from straight_lines1.jpg
src = np.float32([
    [580, 460],  # top left
    [700, 460],  # top right
    [1122, 720],  # bottom right
    [185, 720],  # bottom left
])

# For destination points, I'm choosing points that extend the height of the image, excluding the bonnet region.
dst = np.float32([
    [256, 0],  # top left
    [1050, 0],  # top right
    (1050, 720),  # bottom right
    (256, 720),  # bottom left
])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 256, 0        | 
| 700, 460      | 1050, 0       |
| 1122, 720     | 1050, 720     |
| 185, 720      | 256, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/straight_lines1/plot.jpg)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

If there's no history of detected lanes, or there have been two consecutive sanity check failures, lane lines were detected using a sliding window method (lines ~ 263 in pipeline.py).

If previous lane lines have been detected the it uses the previously calulated polynomial to detected lane pixels on the next iteration. This is show in lines ~ 184 in pipeline.py.

An example of using the **sliding window method**:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/straight_lines2/plot.jpg)

An example of using the **look ahead method**:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/straight_lines2/using_history_plot.jpg)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This is contained in function get\_line\_radius() in `utils.py` and is called in lines 365 and 366 to get the left and right radius, respectively.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in ~ lines 428 in my code in `pipeline.py`.  Here is an example of my result on a test image:

![alt text](https://raw.githubusercontent.com/dpdenton/CarND-Advanced-Lane-Lines/master/output_images/test1/result.jpg)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://github.com/dpdenton/CarND-Advanced-Lane-Lines/blob/master/project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I would probably reduce the threshold of the L channel to help capture more of the white line, and find other ways to reduce the noise of other non-line pixel detected, as occasionally there are a limited number of plots genertated when there's a heavily spaced white line. This is especially true against a light colour road surface where it still struggles to detected lines and is only able to recover because it moves out of that region quickly.

I would develop a 'confidence' metric which would determine the size of the margin used to detect lane pixels, for both the sliding window and look ahead method. The theory being that if the confidence in previously detected lines is high, you should be able to search a relatively small area, as there is a maximum possible change the new line position could be in, based up the speed of the car and the curvature of the line.

If the new line is detected within this region the 'confidence' increases and a smaller region is used next time. Conversly if the region fails to detect an adequate number of pixels the 'confidence' is reduce and the region increases to help capture more possible pixels.

I would also look to improve the sliding window method by ascertaining a value for the width of a line, and using this value to create a window within the window to detect 'line only' pixels, as detecting pixels that go across the whole with of a 100px window is clearly wrong, as the width of a line in pixels in the images is approximately 30px, and can cause the window to go off in an incorrect direction.

