import cv2
import numpy as np
import pickle

def build_squares(img):
#making 10x5 small squares.
#the problem with this approach is that I need to find the optimal value for thresholding
#still working on that
	d = 10
	x, y, w, h = 420, 140, 10, 10 #coordinates of each square
	crop = None
	imgCrop = None
	for i in range(10):
		for j in range(5):
			if np.any(imgCrop == None):
				imgCrop = img[y:y+h, x:x+w]
			else:
				imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
			cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
			x+=w+d
		if np.any(crop == None):
			crop = imgCrop
		else:
			crop = np.vstack((crop, imgCrop)) 
		imgCrop = None
		x = 420
		y+=h+d
	return crop #return the cropped part

def get_hand_hist():
	cam = cv2.VideoCapture(1)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	x, y, w, h = 300, 100, 300, 300
	isfladPressedC, isfladPressedS = False, False
	imgCrop = None
	while True:
		img = cam.read()[1]
		img = cv2.flip(img, 1)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		
		keypress = cv2.waitKey(1)
		if keypress == ord('c'):		
			hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
			isfladPressedC = True
			hist = cv2.calcHist([hsvCrop], [0, 1], None, [210, 230], [0, 210, 2, 230])
			cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
		elif keypress == ord('s'):
			isfladPressedS = True	
			break
		if isfladPressedC:	
			dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 210, 2, 230], 1)
			dst1 = dst.copy()
			disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
			cv2.filter2D(dst,-1,disc,dst)
			blur = cv2.GaussianBlur(dst, (11,11), 0)
			blur = cv2.medianBlur(blur, 15)
			ret,thresh = cv2.threshold(blur,12,255,cv2.THRESH_BINARY)
			thresh = cv2.merge((thresh,thresh,thresh))
			cv2.imshow("Thresh", thresh)
			#cv2.imshow("res", res)
		if not isfladPressedS:
			imgCrop = build_squares(img)
		cv2.imshow("Set hand histogram", img)
	cam.release()
	cv2.destroyAllWindows()
	with open("hist", "wb") as f:
		pickle.dump(hist, f)


get_hand_hist()
