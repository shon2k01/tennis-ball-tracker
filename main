#imports
import cv2 as cv
import numpy as np

def get_source():
    while True:
        choice = input("Please choose input source:\n1.camera\n2.saved video\n")
        if(choice == "1"):return 0
        elif(choice == "2"):
            path = input("Please enter video's full path: ")
            try:
                open(path)
                return path
            except IOError as E:
                print ('File Not Found! Please try again!')

#video source
source = get_source()
videoCapture = cv.VideoCapture(source)

#will be used later for accuracy
prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2 + (y1-y2)**2

#determine boundries of green
greenLower = (20, 50, 140)
greenUpper = (40, 200, 255)

while True:
    #read frames and break if no frame is returned
    ret, frame = videoCapture.read()
    if not ret: break

    #convert to hsv colors for later masking
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    #blurr to cancel kernel noises
    blurred_hsv = cv.GaussianBlur(hsv, (3, 3), cv.BORDER_DEFAULT) #bluring actually made the noise worse ?


	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
    mask = cv.inRange(blurred_hsv, greenLower, greenUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    #returns list of found circles by parameters in form of ((x,y),r)
    circles = cv.HoughCircles(mask.copy(), cv.HOUGH_GRADIENT, 1.1, 1000000, param1 = 60, param2 = 20, minRadius = 10, maxRadius = 1000)

    if circles is not None:

        circles = np.int16(np.around(circles))
        chosen = None

        #each iteration we will choose the "closest" circle to the one found in the previous frame
        #by calculating distances between centers. this makes the program more accurate.
        for i in circles[0,:]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i
        prevCircle = chosen

        #draw a red circle around the ball and a dot at its center
        cv.circle(frame, (chosen[0], chosen[1]), chosen[2],(0,0,255), 3)
        cv.circle(frame, (chosen[0], chosen[1]), 1,(0,0,255), 3)

    #display the results
    cv.imshow("Ball Tracker", frame)

    #quit when 'X' is pressed on the windows
    keyCode = cv.waitKey(1)
    if cv.getWindowProperty("Ball Tracker", cv.WND_PROP_VISIBLE) <1:
        break

#release resources
videoCapture.release()
cv.destroyAllWindows()
