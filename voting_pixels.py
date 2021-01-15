import cv2
import numpy as np


def orientation_mask(frame, norm_quantile=.8):
    # Get the grey-scale part of the image:
    img = frame[:, :, 2]

    gx, gy = np.gradient(img)

    orientation = np.arctan2(gy, gx)
    
    norm = np.hypot(gy, gx)
    threshold = np.quantile(norm, norm_quantile)
    
    mask = np.where(norm > threshold, 1, 0)

    return orientation, norm, mask


def bin2red(mask, orientation):
    """Return the red mask"""
    blue_green = np.where(mask, orientation, 0)
    red = np.where(mask, orientation, 255)
    return np.dstack((blue_green, blue_green, red))


if __name__ == "__main__":
    IMG_TEST = "images/q3_test_image.jpg"

    img = cv2.imread(IMG_TEST)
    hsv_img =  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    orientation, norm, mask = orientation_mask(hsv_img)
    mask_red = bin2red(mask, orientation)

    while True:
        cv2.imshow('Original', img)
        cv2.imshow('Mask', mask_red)
        cv2.imshow('Norm', norm / norm.max().max())
        cv2.imshow('Orientation', orientation)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

    cv2.destroyAllWindows()
