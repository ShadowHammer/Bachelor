import cv2
import os
import cv2
import cv2.xfeatures2d
from matplotlib import pyplot as plt

image_path = os.getcwd() + "/Pothole_coco/test/img-23_jpg.rf.e6aa0daf83e72ccbf1ea10eb6a6ab3bd.jpg"
#print(image_path)
image = cv2.imread(image_path)
gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

SIFT = cv2.SIFT_create()

kp = SIFT.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,image)

cv2.imwrite('sift_keypoints.jpg',img)

img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)

plt.imshow(img),plt.show()


