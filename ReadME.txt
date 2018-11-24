This project despite being a group project was attempted solely by me

Using Python3

Output is shown in the pdf file attached in the zip
I have used two additional image samples than what has been given as test samples
corner_detector.py detects the corner points of the image and returns a corner metric
anms.py does the adaptive non maximal suppression to the passed corner points
feat_desc.py describes the features for each corner point using a 40x40 window having 64 features each
feat_match.py does feature matching using the annoy module in python
est_homograph_ransac estimates the homograpy after implementing Ransac
mymosaic.py returns the stiched image for all the 3 images
utils.py was used to compute gaussian filter for gaussian smoothing

anms is given a 250 return index
Gaussian filter used is 5x5

all the python files and images are saved in the same folder