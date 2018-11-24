
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import numpy as np 
from skimage.feature import corner_harris, corner_subpix, corner_peaks

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def corner_detector(img):
  # Your Code Here
  #cimg = corner_peaks(corner_harris(img), min_distance=5)
  cimg = corner_harris(img)
  return cimg


if __name__ == "__main__":
  img1 = np.array(Image.open('img1.jpg').convert('RGB'))
  img2 = np.array(Image.open('img2.jpg').convert('RGB'))
  img1_gray = rgb2gray(img1)
  img2_gray = rgb2gray(img2)
  im1_corners = corner_detector(img1_gray)
  im2_corners = corner_detector(img2_gray)
  max1 = np.max(np.max(im2_corners))
  print(np.max(im2_corners))
  y,x  = np.where(im2_corners>(0.1*max1))
  #plt.imshow()
  #fig, ax = plt.subplots()
  #ax.imshow(img1, interpolation='nearest', cmap=plt.cm.gray)
  #ax.plot(im1_corners[:, 1], im1_corners[:, 0], '.b', markersize=3)
  fig, ax = plt.subplots()
  ax.imshow(img2, interpolation='nearest', cmap=plt.cm.gray)
  ax.plot(x, y, '.b', markersize=3)

  plt.show()
