
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import numpy as np 
from skimage.feature import corner_harris
from corner_detector import corner_detector
from corner_detector import rgb2gray
from anms import anms

def feat_desc(img, x, y):
  # Your Code Here
  #return descs
  xx,yy = np.meshgrid(np.arange(-15, 21,5), np.arange(-15,21,5))
  #print(xx)
  #print(yy)
  index = np.stack((yy.flatten(), xx.flatten()), axis = -1)
  #print(index.reshape(1,-1,2))
  #print(len(x))
  indexRep = np.tile(index.reshape(1,-1,2),(len(x),1,1))
  #print(indexRep.shape)
  indexCorner = np.stack((y, x), axis = -1).reshape(-1,1,2)
  #print(indexCorner)
  featureIndex = indexCorner + indexRep
  featureIndexY = np.clip(featureIndex[:,:,0], 0, img.shape[0]-1)
  featureIndexX = np.clip(featureIndex[:,:,1], 0, img.shape[1]-1)
  #print(featureIndexY)
  features = img[featureIndexY, featureIndexX]
  featureMean = np.mean(features, axis = 0).reshape(1,-1)
  featuresStd = np.std(features,axis = 0).reshape(1,-1)
  featuresNorm = (features - featureMean)// featuresStd
  #print(featuresNorm)
  fig, ax = plt.subplots()
  ax.imshow(img, interpolation='nearest', cmap = 'gray')
  ax.plot(featureIndexX, featureIndexY, '.b', markersize=1)

  plt.show()
  return featuresNorm.transpose(1,0)





if __name__ == "__main__":
  img1 = np.array(Image.open('2Hill.jpg').convert('RGB'))
  img2 = np.array(Image.open('3Hill.jpg').convert('RGB'))
  img1_gray = rgb2gray(img1)
  img2_gray = rgb2gray(img2)
  im1_corners = corner_detector(img1_gray)
  im2_corners = corner_detector(img2_gray)
  #print(np.min(im1_corners))
  #x1,y1,rmax1 = anms(im1_corners, 250)
  x2,y2, rmax = anms(im2_corners,250)

  # xy = np.load('xy.npy')
  # y = xy[:,1]
  # x = xy[:,0]
  #feat_desc(img1_gray, x1, y1)
  feat_desc(img2_gray, x2, y2)

  # fig, ax = plt.subplots()
  # ax.imshow(img1, interpolation='nearest')
  # ax.plot(x, y, '.b', markersize=3)

  plt.show()

