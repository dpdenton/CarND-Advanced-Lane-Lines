import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

class CameraCalibration(object):

    objpoints = []
    imgpoints = []

    def __init__(self, nx=9, ny=6):

        self.M = None
        self.prev_img = None

        # prepare object points
        self.nx = nx
        self.ny = ny
        self.objp = np.zeros((nx * ny, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    def gen_points(self, img, show=False):

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

        # If found, draw corners
        if ret == True:
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners)

            if show:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
                plt.imshow(img)
                plt.show()

    def undistort(self, img):
        """
        Wrapper for cv2.undistort()
        :param img: img
        :return: cv2.undistort img
        """

        if self.prev_img is not None and self.prev_img.shape == img.shape:
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        else:
            self.calibrate(img)
            return self.undistort(img)

    def calibrate(self, img, save=False):
        """
        Wrapper for cv2.calibrateCamera()
        :param img: img
        :return: None
        """

        img_size = (img.shape[1], img.shape[0])

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints,
            self.imgpoints,
            img_size, None, None
        )

        self.prev_img = img

        if save:
            with open("./pickle/mtx.obj", "wb") as f:
                pickle.dump(self.mtx, f)
            with open("./pickle/dist.obj", "wb") as f:
                pickle.dump(self.dist, f)
            with open("./pickle/prev_img.obj", "wb") as f:
                pickle.dump(self.prev_img, f)

    def load_calibration_mtx(self):

        try:
            with open("./pickle/mtx.obj", "rb") as f:
                self.mtx = pickle.load(f)
            with open("./pickle/dist.obj", "rb") as f:
                self.dist = pickle.load(f)
            with open("./pickle/prev_img.obj", "rb") as f:
                self.prev_img = pickle.load(f)
        except FileNotFoundError:
            return None

        return self.mtx

    def setM(self, src=None, dst=None):
        """
        # Set matrix using points on image "test_images/straight_lines1.jpg"
        :return: self
        """

        # already set
        if self.M is not None:
            return self

        if src is None:
            # For source points I'm grabbing the outer four detected corners
            # hard-coded src points from straight_lines1.jpg
            src = np.float32([
                [580, 460],  # top left
                [700, 460],  # top right
                [1122, 720],  # bottom right
                [185, 720],  # bottom left
            ])

        if dst is None:
            # For destination points, I'm choosing points that extend the height of the image, excluding the bonnet region.
            dst = np.float32([
                [256, 0],  # top left
                [1050, 0],  # top right
                (1050, 720),  # bottom right
                (256, 720),  # bottom left
            ])

        # Given src and dst points, calculate the perspective transform matrix
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

        return self

    def warp(self, img):

        if self.M is not None:
            img_size = (img.shape[1], img.shape[0])
            return cv2.warpPerspective(img, self.M, img_size)
        else:
            self.setM()
            return self.warp(img)

    def test(self, test_fname, save=False):

        test_image = cv2.imread(test_fname)
        dst = self.undistort(test_image)

        fig = plt.figure()

        # before image
        a = fig.add_subplot(1, 2, 1)
        imgplot = plt.imshow(test_image, cmap='gray')
        a.set_title("Before")

        # after image
        a = fig.add_subplot(1, 2, 2)
        imgplot = plt.imshow(dst, cmap='gray')
        a.set_title("After")

        plt.figtext(y=0.1, x=0.1, s="Undistortion of {}".format(test_fname))

        if save:
            plt.savefig("./output_images/undistortion_exmpale_chessboard.jpg")
