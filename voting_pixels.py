import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage, misc
from IPython import display


def orientation_mask(image_filename, threshold=.2, norm_enhancer=10.):
    # reads the image:
    img = cv.imread(image_filename, 0)
    
    gx, gy = np.gradient(img)
    # sobel derivatives
    # gy = cv.Sobel(img, cv.CV_64F, 1, 0)
    # # gx = cv.Sobel(img, cv.CV_64F, 0, 1)
    # gy = np.uint8(np.absolute(gy))
    # gx = np.uint8(np.absolute(gx))

    orientation = np.arctan2(gy, gx)
    # orientation = cv.phase(gy, gx)
    norm = np.hypot(gy, gx) / norm_enhancer

    # magnitude = cv.magnitude(gy, gx)
    mask = np.where(norm > threshold, orientation, 0)

    # while True:
    #     cv.imshow("Original", img)
    #     cv.imshow("gx", gx)
    #     cv.imshow("gy", gy)
    #     cv.imshow("orientation", orientation)
    #     cv.imshow("norm", norm)
    #     cv.imshow("mask", mask)

    #     ch = cv.waitKey(1)
    #     if ch == 27 or ch == ord('q') or ch == ord('Q'):
    #         break
    plt.imshow(orientation, cmap = 'gray')
    plt.show()
    # print(img.shape)

    # display.display(Image.fromarray(img))
    # plt.imshow(img)
    # plt.show()

    # sobel derivatives
    # derivX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    # derivY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    
    # grad = ndimage.sobel(img)

    # while True:
    #     cv.imshow("derivX", derivX)
    #     cv.imshow("derivY", derivY)
    #     cv.imshow("grad", grad)
    #     ch = cv.waitKey(1)
    #     if ch == 27 or ch == ord('q') or ch == ord('Q'):
    #         break

    # # plt.imshow(grad)
    # # plt.show()

    # # orientation and magnitude
    # orientation = derivX + derivY
    # # cv.phase(derivX, derivY)
    # magnitude = cv.magnitude(derivX, derivY)
    # # orientation = 0
    # # magnitude = 0
    # # mask = 0

    # _, mask = cv.threshold(magnitude, threshold, 255, cv.THRESH_BINARY)

    return orientation, norm, mask


if __name__ == "__main__":
    IMG_TEST = "q3_test_image.jpg"

    orientation, norm, mask = orientation_mask(IMG_TEST)
