#identify and track any orange colors using opencv

#import cv2
import cv2
#import numpy
import numpy

#access webcam
input=cv2.VideoCapture(0)

#infinite loop
while(1):
	
	#read incoming video feed
	_, frame=input.read()
	#convert from BGR to HSV
	hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	#Lower limit of orange color that will be recognized
	lower_orange=numpy.array([0,110,110])
	#Upper limit of orange color that will be recognized
	upper_orange=numpy.array([15,255,255])
	
	#mask show orange objects in white and everything else in black
	mask=cv2.inRange(hsv, lower_orange, upper_orange)
	
	#Clean up output
	element=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
	mask=cv2.erode(mask, element, iterations=2)
	mask=cv2.dilate(mask, element, iterations=2)
	mask=cv2.erode(mask, element)

	#Creates a red rectangle around any orange objects
	contours, hierarchy=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	

	#Creates a box around the maximum area
	maximumArea=0
	bestContour=None
	for contour in contours:
		currentArea=cv2.contourArea(contour)
		if currentArea>maximumArea:
			bestContour=contour
			maximumArea=currentArea
	
	
	if bestContour is not None:
		#(x,y) is the center point of the rectangle; (w,h) is its width and height
		x, y, w, h=cv2.boundingRect(bestContour)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
		

	#show the unedited video feed
	cv2.imshow('frame', frame)
	#show the feed with the mask applied (all orange objects appear white, everything else black)
	cv2.imshow('mask', mask)
	
	#press Escape key to exit mask and video feed windows
	k=cv2.waitKey(5) &0xFF
	if k==27:
		break

cv2.destroyAllWindows()
