
import random
from est_homography import est_homography
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import numpy as np 
from skimage.feature import corner_harris
from corner_detector import corner_detector
from corner_detector import rgb2gray
from feat_desc import feat_desc
from annoy import AnnoyIndex
from scipy import signal
from utils import GaussianPDF_2D
from matplotlib.patches import ConnectionPatch
from feat_match import feat_match
from anms import anms


def ransac_est_homography(x1, y1, x2, y2, thresh):
  # Your Code Here
  index = np.arange(0, len(x1)).tolist()
  index_2_Vstacked = np.vstack((x2,y2,np.ones(len(x1))))
  inlierCur = 0
  index_1_Vstacked = np.vstack((x1,y1,np.ones(len(x1))))
  for i in range(1000):
    indexRand = np.array(random.sample(index, 4))
    x1Rand = x1[indexRand]
    x2Rand = x2[indexRand]

    y1Rand = y1[indexRand]
    y2Rand = y2[indexRand]
    

    Homograph = est_homography(x1Rand, y1Rand, x2Rand, y2Rand)
    #print(Homograph)
    
    newIndex = np.dot(Homograph, index_1_Vstacked)
    newIndexNorm = newIndex/ newIndex[-1,:]
      #print(newIndexNorm)

    errorMetric = np.square(newIndexNorm - index_2_Vstacked)
      #print('square',errorMetric)
    errorMetric1 = np.sum(errorMetric, axis = 0)
      #print('sum',errorMetric)
    errorMetric2 = np.sqrt(errorMetric1)

    inliers = len(errorMetric2[errorMetric2 < thresh])
    #print('inliers', inliers)

    if inliers > inlierCur:
      inlierCur = inliers
      HomographFinal = Homograph
      inlier_ind = (errorMetric2 < thresh).astype(int)
      #print('sum',errorMetric1)
      #print('sqrt',errorMetric2)


      #print('inliersFinal',inliers)

  H = HomographFinal
  #print(inlier_ind)
  #print(H)
  #H = 1



  return H, inlier_ind



if __name__ == "__main__":
  img1 = np.array(Image.open('1L.jpg').convert('RGB'))
  img2 = np.array(Image.open('1M.jpg').convert('RGB'))
  img1_gray = rgb2gray(img1)
  img2_gray = rgb2gray(img2)
  im1_corners = corner_detector(img1_gray)
  im2_corners = corner_detector(img2_gray)
  #print(np.min(im1_corners))
  Gaus = GaussianPDF_2D(0,0.5,5,5)
  img1_gray = signal.convolve2d(img1_gray, Gaus, 'same')
  img2_gray = signal.convolve2d(img2_gray, Gaus, 'same')



  x1,y1,rmax1 = anms(im1_corners, 250)
  x2,y2, rmax = anms(im2_corners,250)

  # xy = np.load('xy.npy')
  # y = xy[:,1]
  # x = xy[:,0]
  feat_desc(img1_gray, x1, y1)
  feat_desc(img2_gray, x2, y2)

  feat1 = feat_desc(img1_gray, x1, y1)
  feat2 = feat_desc(img2_gray, x2, y2)
  #print(feat1.shape)

  feat_match_ind = feat_match(feat1, feat2)
  print(len(feat_match_ind[feat_match_ind!=-1]))
  a = np.arange(0, len(x1))

  #x1 = x1[a[feat_match_ind != -1]]
  #y1 = y1[a[feat_match_ind != -1]]
  #x2 = x2[feat_match_ind[feat_match_ind != -1]]
  #y2 = y2[feat_match_ind[feat_match_ind != -1]]
  x1 = np.load('x1.npy')
  x2 = np.load('x2.npy')

  y1 = np.load('y1.npy')
  y2 = np.load('y2.npy')

  Hfinal, inlier_ind = ransac_est_homography(x1, y1, x2, y2, 0.5)

  x1Best = x1[inlier_ind > 0]
  y1Best = y1[inlier_ind > 0]

  x2Best = x2[inlier_ind > 0]
  y2Best = y2[inlier_ind > 0]

  x1Worst = x1[inlier_ind == 0]
  y1Worst = y1[inlier_ind == 0]

  x2Worst = x2[inlier_ind == 0]
  y2Worst = y2[inlier_ind == 0]
  
  

  f = plt.figure(figsize=(10,10))
  ax1 = f.add_subplot(121)
  ax2 = f.add_subplot(122)

  ax1.imshow(img1)
  ax1.axis('off')
  ax2.imshow(img2)
  ax2.axis('off')

  #output_match = output_match.squeeze()
  #output_match = output_match.astype(int)

  ax1.plot(x1Best, y1Best, 'ro', markersize=3)
  ax2.plot(x2Best, y2Best, 'ro', markersize=3)
  ax1.plot(x1Worst, y1Worst, 'bo', markersize=5)
  ax2.plot(x2Worst, y2Worst, 'bo', markersize=5)

  
  for i in range(x1Best.shape[0]):
    con = ConnectionPatch(xyA=(x2Best[i], y2Best[i]), xyB=(x1Best[i], y1Best[i]), coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="red")
    ax2.add_artist(con)
  for i in range(x1Worst.shape[0]):
    con = ConnectionPatch(xyA=(x2Worst[i], y2Worst[i]), xyB=(x1Worst[i], y1Worst[i]), coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="blue")
    ax2.add_artist(con)


  # fig, ax = plt.subplots()
  # ax.imshow(img, interpolation='nearest')
  # ax.plot(x.flatten(), y.flatten(), '.r', markersize=1)
  plt.show()

  # fig, ax = plt.subplots()
  # ax.imshow(img, interpolation='nearest')
  # ax.plot(x.flatten(), y.flatten(), '.r', markersize=1)
  plt.axis('off')
  plt.show()
  #print(im1MatchCorner, im1MatchCorner.shape)
  #print(im2MatchCorner, im1MatchCorner.shape)
