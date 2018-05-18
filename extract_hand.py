#extracting the background
#this is called foreground extraction or background reduction
import numpy as np
import cv2
#
#cap = cv2.VideoCapture(1)
#fgbg = cv2.createBackgroundSubtractorMOG2()
#
#while(1):
#    ret, frame = cap.read()#read frame
#
#    fgmask = fgbg.apply(frame)  #apply the mask to the frame
# 
#    cv2.imshow('fgmask',frame)
#    cv2.imshow('frame',fgmask)
#
#    
#    k = cv2.waitKey(30) & 0xff
#    if k == 27:
#        break
#    
#
#cap.release()
#cv2.destroyAllWindows()


img = cv2.imread(r'C:\Users\Karim El Guermai\Desktop\PROGRAMMING\AllData\LoicMarieMini_A\train\A\A261.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
#retval, threshold = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1 )
#255 is hte maxiumum threshold, the more we have the light the more we need to choose a biger value (90)
cv2.imshow('original',img)
cv2.imshow('threshold',th)
cv2.waitKey(0)
cv2.destroyAllWindows()
