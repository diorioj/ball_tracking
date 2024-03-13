from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import csv
import serial

# configure arguments
# -h/--help is automatically generated
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=5, help="max buffer size/tail length")
ap.add_argument("-c", "--camera", type=int, default=0, help="camera device index")
ap.add_argument("-r", "--radius", type=int, default=5, help="minimum radius of target")
ap.add_argument("-w", "--width", type=int, default=600, help="set video resize width")
ap.add_argument("-o", "--output", default='tracking_output.csv', help="output file name")
ap.add_argument("-V", "--verbose", action='store_true', help="output coordinates of target to command line") 
ap.add_argument("-u", "--uart", default='/dev/ttyUSB1', help="path to the uart serial device")
args = vars(ap.parse_args())

if args.get("uart", True):
	# UART with pyserial
	ser = serial.Serial(args["uart"], 9600, timeout=1)
	ser.reset_input_buffer()

# define the lower and upper boundaries
# of the target balls in the HSV color space
colors = ["green", "pink", "yellow"]
bgr = [(50, 205, 50), (147, 105, 255), (0, 255, 255)]

greenBounds = ((60, 40, 40) , (80, 255, 255)) # (lower, upper)
greenBuffer = deque(maxlen=args["buffer"]) # points to be visualized as "tail"

pinkBounds = ((150, 80, 80), (170, 255, 255))
pinkBuffer = deque(maxlen=args["buffer"])

yellowBounds = ((20, 60, 60), (40, 255, 255))
yellowBuffer = deque(maxlen=args["buffer"])

boundaries = [greenBounds, pinkBounds, yellowBounds]
buffers = [greenBuffer, pinkBuffer, yellowBuffer]

# initialize video source and get frame center after resizing
if not args.get("video", False):
	# webcam stream
	vs = VideoStream(src=args["camera"]).start()
	frameCenter = (np.array(imutils.resize(vs.read(), width=args["width"]).shape[:2])//2)[::-1]
else:
	# video file
	vs = cv2.VideoCapture(args["video"])
	frameCenter = (np.array(imutils.resize(vs.read()[1], width=args["width"]).shape[:2])//2)[::-1]

# allow the camera or video file to warm up
time.sleep(2.0)

# initialize list of data points to be saved to csv or xmitted
data = []

# keep looping until video source stops
while True:

	# grab the current frame
	frame = vs.read()
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame

	# we have reached the end of the video if there is no frame
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV color space
	frame = imutils.resize(frame, width=args["width"])
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	masks = [None] * len(colors)
	frameData = [[0, 0]] * len(colors)

	for i in range(len(colors)):
		masks[i] = cv2.inRange(hsv, boundaries[i][0], boundaries[i][1])
		masks[i] = cv2.erode(masks[i], None, iterations=2)
		masks[i] = cv2.dilate(masks[i], None, iterations=2)

		# find contours in the mask and initialize the current (x, y) center of the ball
		contours = cv2.findContours(masks[i].copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		contours = imutils.grab_contours(contours)
		center = None

		# only proceed if at least one contour was found
		if len(contours) > 0:
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			c = max(contours, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
			M = cv2.moments(c)
			center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

			# save dx, dy of center of contour from center of frame to data list
			frameData[i] = center
			
			# update the buffer queue
			buffers[i].appendleft(center)

			# only draw circle if the radius meets a minimum size
			if radius > args["radius"]:
				# draw the circle and centroid on the frame
				cv2.circle(frame, (int(x), int(y)), int(radius),
					bgr[i], 2)
				cv2.circle(frame, center, int(args["radius"]), bgr[i], -1)

	xSum = 0
	ySum = 0
	count = 0
	for x, y in frameData:
		if (x > 0 or y > 0):
			count = count + 1
			xSum = xSum + x
			ySum = ySum + y
			
	if count > 0:
		# draw circle for middle of 3 balls
		triCenter = [(xSum//count), (ySum//count)]
		cv2.circle(frame, triCenter, int(args["radius"]), (0, 0, 0), -1)
		data.append([np.subtract(triCenter, frameCenter), count])
	
		output = "(dx: " + str(data[-1][0][0]) + ", dy: " + str(data[-1][0][1]) + "), count: " + str(data[-1][1])

		if args.get("uart", True):
			# tx/rx arduino
			ser.write((output + "\n").encode('utf-8'))
			line = ser.readline().decode('utf-8').rstrip()

			if args.get("verbose", True):
				print(line)

		# if verbose, output data to command line
		elif args.get("verbose", True):
			print(output)

	# draw tails
	# loop over the set of tracked points
	for j in range(len(colors)):
		for i in range(1, len(buffers[j])):
			# if either of the tracked points are None, ignore
			# them
			if buffers[j][i - 1] is None or buffers[j][i] is None:
				continue
			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
			hl, sl, vl = boundaries[j][0]
			cv2.line(frame, buffers[j][i - 1], buffers[j][i], bgr[j], thickness)

	# show the frame to our screen
	cv2.imshow("Ball Tracking", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	print("Stopping video stream...")
	vs.stop()
	time.sleep(3)
# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()

# output data to csv file
print("Writing data to " + args["output"] + "...")
with open(args["output"], 'w', newline='') as f:
	writer = csv.writer(f)
	writer.writerows(data)
