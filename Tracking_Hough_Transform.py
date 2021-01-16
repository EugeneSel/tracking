import numpy as np
import cv2
from voting_pixels import orientation_mask, bin2red
roi_defined = False
 
def define_ROI(event, x, y, flags, param):
	global r, c, w, h, roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2 - r)
		w = abs(c2 - c)
		r = min(r, r2)
		c = min(c, c2)  
		roi_defined = True


def get_index(value, interval_size=np.pi):
    index = (value + np.pi) // interval_size

    # Consider the board effect:
    return index if value < np.pi else index - 1


cap = cv2.VideoCapture('Sequences/Antoine_Mug.mp4')

# take first frame of the video
ret, frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r, c), (r + h, c + w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break
 
track_window = (r, c, h, w)
# set up the ROI for tracking
roi = frame[c:c + w, r:r + h]
# conversion to Hue-Saturation-Value space
# 0 < H < 180; 0 < S < 255; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
orientation, norm, mask = orientation_mask(hsv_roi, norm_quantile=.8)
# generate R-table:
r_table = {}
omega = (r + h // 2, c + w // 2)

# Define number of orientations:
n_orientations = 90
interval_size = 2 * np.pi / n_orientations

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if mask[i, j]:
            idx = str(get_index(orientation[i, j], interval_size))
            if idx in r_table:
                r_table[idx] += [(omega[0] - j - r, omega[1] - i - c)]
            else:
                r_table[idx] = [(omega[0] - j - r, omega[1] - i - c)]

# Setup the termination criteria: either 10 iterations,
# or move by less than 1 pixel
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

cpt = 1
while True:
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        orientation, norm, mask = orientation_mask(hsv, norm_quantile=.9)
        t_hough = np.zeros_like(mask)

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    idx = str(get_index(orientation[i, j], interval_size))
                    if idx in r_table:
                        for v in r_table[idx]:
                            if j + v[0] >= 0 and j + v[0] < t_hough.shape[1] and i + v[1] >= 0 and i + v[1] < t_hough.shape[0]:
                                t_hough[i + v[1], j + v[0]] += 1.

        # Argmax:
        center_y, center_x = np.unravel_index(np.argmax(t_hough), t_hough.shape)
        r, c = max(center_x - h // 2, 0), max(center_y - w // 2, 0)

        # Mean Shift:
        # ret, track_window = cv2.meanShift(t_hough / t_hough.max().max(), track_window, term_crit)
        # r, c, h, w = track_window

        # Draw a blue rectangle on the current image
        frame_tracked = cv2.rectangle(frame, (r, c), (r + h, c + w), (255, 0, 0), 2)
        mask_red = bin2red(mask, orientation)

        cv2.imshow('Sequence', frame_tracked)
        cv2.imshow('Mask', mask_red)
        cv2.imshow('Norm', norm / norm.max().max())
        cv2.imshow('Orientation', orientation)
        cv2.imshow('Hough Transform', t_hough / t_hough.max().max())

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png'%cpt, frame_tracked)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
