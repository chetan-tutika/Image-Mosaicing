
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
from ransac_est_homography import ransac_est_homography
from scipy.ndimage import geometric_transform
from scipy import ndimage
from numpy.linalg import inv
from anms import anms

def frame(img, img1, H, ValueB):
  imgC = img1.copy()
  imgC = imgC.astype(float)
  xx,yy = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0,img.shape[0]))
  index = np.stack((xx.flatten(), yy.flatten()), axis = -1).transpose(1,0)
  index = np.vstack((index, np.ones(index.shape[1])))
  #print('index shape',index.shape)

  #print(index)
  newIndex = np.matmul(H, index)
  newIndex = (newIndex/newIndex[-1,:])
  xNew = newIndex[0,:]# - np.min(newIndex[0,:])).astype(int)#,0, img1.shape[1]-1)
  yNew = newIndex[1,:]# - np.min(newIndex[1,:])).astype(int)#,0, img1.shape[0]-1)

  xNewMin = np.min(xNew)
  yNewMin = np.min(yNew)
  xNewMax = np.max(xNew)
  yNewMax = np.max(yNew)
  #print('Xmin', xNewMin)
  #print('Xmin', np.min(xNew))
  yShift, xShift = 0, 0
  if yNewMin < 0:
    imgC = np.pad(imgC, ((-int(yNewMin),0),(0,0),(0,0)),'constant', constant_values = 0)
    yShift =np.abs(yNewMin)

  if yNewMax > img1.shape[0]-1:
    imgC = np.pad(imgC, ((0,int(yNewMax - img1.shape[0])),(0,0),(0,0)),'constant', constant_values =0)
  if xNewMin < 0:
    imgC = np.pad(imgC, ((0,0),(-int(xNewMin),0),(0,0)),'constant', constant_values =0)
    xShift = np.abs(xNewMin)
  if xNewMax > img1.shape[1]-1:
    imgC = np.pad(imgC, ((0,0),(0,int(xNewMax - img1.shape[1])),(0,0)), 'constant', constant_values =0)
  plt.figure(num='pad')
  plt.imshow(imgC)
  #plt.show()
  #print('index', index)
  
  TransMat = np.array([[1,0,xShift],[0,1, yShift],[0,0,1]])
  newH = np.matmul(TransMat ,H)

  newIndexT = np.matmul(newH, index)
  newIndexT = (newIndexT/newIndexT[-1,:])
  xNewT = newIndexT[0,:]# - np.min(newIndex[0,:])).astype(int)#,0, img1.shape[1]-1)
  yNewT = newIndexT[1,:]



  
  frSizex = (np.max(xNewT) - np.min(xNewT) + 1).astype(int)
  frSizey = (np.max(yNewT) - np.min(yNewT) + 1).astype(int)

  #print('frrrr',frSizey, frSizex)

  frame = np.zeros((frSizey, frSizex,3), dtype = 'float')
  print('frame shape', frame.shape[0],frame.shape[1])
  xxZ,yyZ = np.meshgrid(np.arange(np.min(xNewT), np.max(xNewT)), np.arange(np.min(yNewT), np.max(yNewT)))
  print('index shape',xxZ.shape[0],xxZ.shape[1])
  indexZ = np.stack((xxZ.flatten(), yyZ.flatten()), axis = -1).transpose(1,0)
  indexZ = np.vstack((indexZ, np.ones(indexZ.shape[1])))

  hInv = inv(newH)
  invIndexZ = np.matmul(hInv, indexZ)
  invIndexZ = (invIndexZ/invIndexZ[-1,:]).astype(int)
  xInv = invIndexZ[0,:]# - np.min(newIndex[0,:])).astype(int)#,0, img1.shape[1]-1)
  #print('xinv shape',xInv.shape)
  yInv = invIndexZ[1,:]

  xmin = int(np.min(xNewT))
  xmax =  int(np.max(xNewT))

  ymin = int(np.min(yNewT)) 
  ymax = int(np.max(yNewT))

  frame[:,:,0] = ndimage.map_coordinates(img[:,:,0], [yInv, xInv], mode = 'constant', order=1).reshape(frame.shape[0], frame.shape[1])
  frame[:,:,1] = ndimage.map_coordinates(img[:,:,1], [yInv, xInv], mode = 'constant', order=1).reshape(frame.shape[0], frame.shape[1])
  frame[:,:,2] = ndimage.map_coordinates(img[:,:,2], [yInv, xInv], mode = 'constant', order=1).reshape(frame.shape[0], frame.shape[1])
  #print('llll',imgC.shape, frame.shape)
  plt.figure(num = 'frame')
  plt.imshow(frame)

  # print('',ymin, ymax, xmin, xmax)
  # print(frame.shape)
  # print(imgC.shape)

  imgC[ymin:ymin+frame.shape[0]-3, xmin:xmin+frame.shape[1]-2,:] += frame[:frame.shape[0]-3,:frame.shape[1]-2,:]
  #imgC = 0.5*imgC
  pH = min(frame.shape[0], imgC.shape[0])
  #print('pH',pH)

  if ValueB is True:
    ow = img1.shape[1] - xmin
    ow = int(ow)
    #mask = frame[int(yShift):int(yShift+img1.shape[0]), 0:int(ow), :] > 0
    mask = np.logical_and(frame[:pH, 0:ow, :] > 0, imgC[:pH, xmin:img1.shape[1], :] > 0)
    mask = mask.astype(float)
    mask[mask==1] = 0.5
    mask[mask==0] = 1

    mask_H, mask_W, _ = mask.shape

    imgC[:mask_H, xmin:int(xmin+mask_W), :] *= mask
  if ValueB is False:
    ow = xShift
    ow = int(ow)
    #mask = frame[int(yShift):int(yShift+img1.shape[0]), int(ow):, :] > 0
    mask = np.logical_and(frame[:pH, ow:, :] > 0, imgC[:pH, ow:frSizex, :] > 0)
    mask = mask.astype(float)
    mask[mask==1.0] = 0.5
    mask[mask==0.0] = 1

    mask_H, mask_W, _ = mask.shape

    imgC[:mask_H, int(ow):int(ow+mask_W), :] *= mask    

  plt.figure()
  imgC = imgC.astype('uint8')
  plt.imshow(imgC)
  plt.show()



  return frame, imgC


def stitch(img1, img2, Value):
  img1_gray = rgb2gray(img1)
  img2_gray = rgb2gray(img2)
  Gaus = GaussianPDF_2D(0.1,1,1,1)


  im1_corners = corner_detector(img1_gray)
  im2_corners = corner_detector(img2_gray)

  img1_gray = signal.convolve2d(img1_gray, Gaus, 'same')
  img2_gray = signal.convolve2d(img2_gray, Gaus, 'same')
  x1,y1,rmax1 = anms(im1_corners, 250)
  x2,y2,rmax2 = anms(im2_corners, 250)


  feat1 = feat_desc(img1_gray, x1, y1)
  feat2 = feat_desc(img2_gray, x2, y2)
  a = np.arange(0, len(x1))



  feat_match_ind12 = feat_match(feat1, feat2)
  x2_match1 = x1[a[feat_match_ind12 != -1]]
  y2_match1 = y1[a[feat_match_ind12 != -1]]
  print('MatchFeat', len(feat_match_ind12[feat_match_ind12!=-1]))

  x1_match2 = x2[feat_match_ind12[feat_match_ind12 != -1]]
  y1_match2 = y2[feat_match_ind12[feat_match_ind12 != -1]]

  Hfinal21, inlier_ind21 = ransac_est_homography(x2_match1, y2_match1, x1_match2, y1_match2, 0.5)


  #print('inliers',len(inlier_ind21[inlier_ind21==1]))
  frame1, imgC1 = frame(img1,img2, Hfinal21, Value)

  return imgC1, frame1






def mymosaic(img_input):
  # Your Code Here
  img1 = img_input[0]
  img2 = img_input[1]
  img3 = img_input[2]

  ImgC1, frame1 = stitch(img1, img2, False)
  
  img_mosaic, frame2 = stitch(img3, ImgC1, True)

  return img_mosaic



if __name__ == "__main__":
  img1 = np.array(Image.open('1Hill.jpg').convert('RGB'))
  img2 = np.array(Image.open('2Hill.jpg').convert('RGB'))
  img3 = np.array(Image.open('3Hill.jpg').convert('RGB'))

  imglist = list([img1, img2, img3])
  finalMos = mymosaic(imglist)

  
  plt.imshow(finalMos)

  #plt.figure()
  #plt.imshow(frame2)
  plt.show()

 