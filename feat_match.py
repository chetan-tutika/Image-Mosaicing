
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
from anms import anms

def feat_match(descs1, descs2):
  # Your Code Here
  fD1, NoP1 = descs1.shape[:2]
  fD2, NoP2 = descs2.shape[:2]
  l = []
  t = AnnoyIndex(fD1, metric ="euclidean")

  for i in range(NoP2):
    t.add_item(i,descs2[:,i])
  t.build(50)

  featMatchIndex = np.zeros((NoP1), dtype = int)

  for i in range(NoP1):
    uidx, dist = t.get_nns_by_vector(descs1[:,i], 2, include_distances=True)
    if dist[0]/dist[1] < 0.6:
      featMatchIndex[i] =  uidx[0]
    else:
      featMatchIndex[i] = -1


  return featMatchIndex


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




  #print(np.min(im1_corners))
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

  x1 = x1[a[feat_match_ind != -1]]
  y1 = y1[a[feat_match_ind != -1]]
  x2 = x2[feat_match_ind[feat_match_ind != -1]]
  y2 = y2[feat_match_ind[feat_match_ind != -1]]
  

  #print(im1MatchCorner, im1MatchCorner.shape)
  #print(im2MatchCorner, im1MatchCorner.shape)

  f = plt.figure(figsize=(10,10))
  ax1 = f.add_subplot(121)
  ax2 = f.add_subplot(122)

  ax1.imshow(img1)
  ax1.axis('off')
  ax2.imshow(img2)
  ax2.axis('off')

  ax1.plot(x1, y1, 'bo', markersize=3)
  ax2.plot(x2, y2, 'bo', markersize=3)

  
  for i in range(len(x1)):
    #print(i)
    con = ConnectionPatch(xyA=(x2[i], y2[i]), xyB=(x1[i], y1[i]), coordsA="data", coordsB="data", axesA=ax2, axesB=ax1, color="red")
    ax2.add_artist(con)

  # fig, ax = plt.subplots()
  # ax.imshow(img, interpolation='nearest')
  # ax.plot(x.flatten(), y.flatten(), '.r', markersize=1)
  plt.show()

