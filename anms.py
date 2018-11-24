
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import numpy as np 
from skimage.feature import corner_harris
from corner_detector import corner_detector

def rgb2gray(I_rgb):
	r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
	I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
	return I_gray
def anms(cimg, max_pts):
  xx,yy = np.meshgrid(np.arange(0, cimg.shape[1]), np.arange(0,cimg.shape[0]))
  index = np.stack((yy.flatten(), xx.flatten()), axis = -1)
  cimg_v = cimg.reshape((-1,1))
  cimg_h = cimg.reshape(1,-1)
  maxValue = np.zeros((cimg_v.shape[0],1))
  '''
  for i in range(cimg.shape[0]*cimg.shape[1]):
    #print(i)
    cimgC = cimg.copy()
    cimgC[cimgC == cimg_h[0,i]] = np.min(cimg) - 999.0

    indexofStrong = np.array(np.argwhere(cimgC > 0.9*cimg_v[i,0]))
    #indexy = indexOfStrong[0,:]
    #indexx = indexofStrong[1,:]
    #print(indexofStrong.shape[0])
    sourceIndex = np.array(np.argwhere(cimg == cimg_h[0,i]))
    #sourceIndexX = sourceIndex[1]
    #sourceIndexY = sourceIndexY[0]
    sourceIndexRep = np.tile(sourceIndex,(indexofStrong.shape[0],1))
    #print('show',sourceIndexRep.shape)
    subt_sq = np.square(sourceIndexRep - indexofStrong)
    #print(subt_sq.shape)
    maxValue[i,0] = np.min(np.sqrt(np.sum(subt_sq, axis = 1)))
  '''
  coord1 = np.argwhere(cimg > np.median(cimg) + cimg.std()/3)
  coord = coord1.reshape(-1,1,2)
  print(coord.shape)
  coordTrans = coord.transpose((1,0,2))
  print(coordTrans.shape)
  coordVStacked = np.tile(coordTrans,(coord.shape[0],1,1))
  print(coordVStacked.shape)
  k = np.subtract(coord, coordVStacked)
  print('s')
  coordDistsq = np.square(k)
  print('s1')
  coordDistsum = np.sum(coordDistsq, axis = 2)
  print('s2')
  coordDist = np.sqrt(coordDistsum)
  coordDist[coordDist == 0] = np.inf 

  cimgStrong = cimg[coord1[:,0], coord1[:,1]].reshape(-1,1)
  cimgStrongStack = np.tile(cimgStrong.reshape(1,-1), (cimgStrong.shape[0],1))
  cimgStrongMax = cimgStrongStack - 0.9*cimgStrong
  coordDist[cimgStrongMax<0] = np.inf
  distV = np.min(coordDist, axis = 1)
  maxDist = np.sort(-distV)[0]
  indexMinDist = np.argsort(-distV)[:max_pts]
  coordFinal = coord[indexMinDist,0]




  print(coordVStacked.shape)
  print(coordDist.shape)
  # np.save(coordFinal, name)





  #print(maxValue)

    


  #cimg_hRep = np.tile(cimg_h,(cimg_v.shape[0], 1))
  #maxValPerPixel = np.amax((cimg_v - cimg_h), axis = 1)
  #print(cimg.shape[0]*cimg.shape[1])



  #print(yy)

  return coordFinal[:,1], coordFinal[:, 0], maxDist

if __name__ == "__main__":
  img1 = np.array(Image.open('1M.jpg').convert('RGB'))
  img2 = np.array(Image.open('img2.jpg').convert('RGB'))
  img1_gray = rgb2gray(img1)
  img2_gray = rgb2gray(img2)
  im1_corners = corner_detector(img1_gray)
  im2_corners = corner_detector(img2_gray)
  print(np.min(im1_corners))

  x1,y1,rmax1 = anms(im1_corners, 250)
  x2,y2, rmax = anms(im2_corners,250)

  fig, ax = plt.subplots()
  ax.imshow(img1, interpolation='nearest')
  ax.plot(x1, y1, '.b', markersize=3)

  fig, ax = plt.subplots()
  ax.imshow(img2, interpolation='nearest')
  ax.plot(x2, y2, '.b', markersize=3)

  plt.show()
  '''



  print(np.max(im2_corners))
  y1,x1  = np.where(im1_corners>(0.1*np.max(im1_corners)))
  y2,x2 = np.where(im2_corners>(0.1*np.max(im2_corners)))
  #plt.imshow()
  #print(np.median(im1_corners))
  fig, ax = plt.subplots()
  ax.imshow(img1, interpolation='nearest', cmap=plt.cm.gray)
  ax.plot(x1, y1, '.b', markersize=3)
  fig, ax = plt.subplots()
  ax.imshow(img2, interpolation='nearest', cmap=plt.cm.gray)
  ax.plot(x2, y2, '.b', markersize=3)
  plt.show()
  '''