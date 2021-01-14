import numpy as np

def orientation_mask(frame, norm_quantile=.5):
    # reads the image:
    # img = cv.imread(image_filename, 0)
    img = frame[:, :, 2]
    gx, gy = np.gradient(img)
    # print(gx, gy)
    # sobel derivatives
    # gy = cv.Sobel(img, cv.CV_64F, 1, 0)
    # # gx = cv.Sobel(img, cv.CV_64F, 0, 1)
    # gy = np.uint8(np.absolute(gy))
    # gx = np.uint8(np.absolute(gx))

    orientation = np.arctan2(gy, gx)
    # orientation = cv.phase(gy, gx)
    norm = np.hypot(gy, gx)
    threshold = np.quantile(norm, norm_quantile)

    # magnitude = cv.magnitude(gy, gx)
    mask = np.where(norm > threshold, 1, 0)
    # print(mask)
    # plt.subplot(2,2,1)
    # plt.imshow(orientation, cmap = 'gray')
    # plt.subplot(2,2,2)
    # plt.imshow(norm, cmap = 'black')
    # plt.show()
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
