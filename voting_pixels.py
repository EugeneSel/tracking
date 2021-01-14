import numpy as np

def orientation_mask(frame, norm_quantile=.5):
    # Get the grey-scale part of the image:
    img = frame[:, :, 2]

    gx, gy = np.gradient(img)

    orientation = np.arctan2(gy, gx)
    
    norm = np.hypot(gy, gx)
    threshold = np.quantile(norm, norm_quantile)
    
    mask = np.where(norm > threshold, 1, 0)

    return orientation, norm, mask


# if __name__ == "__main__":
#     IMG_TEST = "q3_test_image.jpg"

    #orientation, magnitude, mask = orientation_mask(IMG_TEST)

    # while True:
    #     cv.imshow("orientation", orientation)
    #     cv.imshow("magnitude", magnitude)
    #     cv.imshow("mask", mask)

    #     ch = cv.waitKey(1)
    #     if ch == 27 or ch == ord('q') or ch == ord('Q'):
    #         break

    # cv.destroyAllWindows()
