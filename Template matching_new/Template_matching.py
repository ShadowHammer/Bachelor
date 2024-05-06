import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

image_path = os.getcwd()
images = []
image_number = 10
for i in range(1,image_number):
   images.append(cv.imread(image_path +'\\pothole_' + str(i) + ".jpg", cv.IMREAD_GRAYSCALE))

template = cv.imread(image_path + '\\template.jpg', cv.IMREAD_GRAYSCALE)


w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

for j in range(len(images)):
   i = 1
   for meth in methods:
      
      method = eval(meth)

      # Apply template Matching
      res = cv.matchTemplate(images[j], template, method)
      min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
      print(meth)
      print(cv.minMaxLoc(res))

      # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
      if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
         top_left = min_loc
      else:
         top_left = max_loc
         bottom_right = (top_left[0] + w, top_left[1] + h)
      
      plt.subplot(2,len(methods),i),plt.imshow(res,cmap = 'gray')
      if meth == "cv.TM_CCOEFF":
         plt.title("MR_CCOEFF")
      elif meth == "cv.TM_CCOEFF_NORMED":
         plt.title("MR_CCOEFF_NORMED")
      elif meth == "cv.TM_CCORR":
         plt.title("MR_CCORR")
      elif meth == "cv.TM_CCORR_NORMED":
         plt.title("MR_CCORR_NORMED")
      elif meth == "cv.TM_SQDIFF":
         plt.title("MR_SQDIFF")
      elif meth == "cv.TM_SQDIFF_NORMED":
         plt.title("MR_SQDIFF_NORMED")

      plt.axis('off')
      i += 1
      img_with_rect = cv.cvtColor(images[j], cv.COLOR_GRAY2RGB)
      cv.rectangle(img_with_rect, top_left, bottom_right, (0, 255, 0), 2)

      plt.subplot(2, len(methods), i)
      plt.imshow(img_with_rect)
      if meth == "cv.TM_CCOEFF":
         plt.title("CCOEFF")
      elif meth == "cv.TM_CCOEFF_NORMED":
         plt.title("CCOEFF_NORMED")
      elif meth == "cv.TM_CCORR":
         plt.title("CCORR")
      elif meth == "cv.TM_CCORR_NORMED":
         plt.title("CCORR_NORMED")
      elif meth == "cv.TM_SQDIFF":
         plt.title("SQDIFF")
      elif meth == "cv.TM_SQDIFF_NORMED":
         plt.title("SQDIFF_NORMED")
      plt.axis('off')
      i += 1
   
   plt.show()
   
      

      