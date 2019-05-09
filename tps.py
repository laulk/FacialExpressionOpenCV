import cv2
import numpy as np
#import matplotlib.pyplot as plt
import sys
import dlib
import functools
import imutils
from imutils import face_utils
import math


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def readLandmarks(image) :
    image = cv2.imread(image)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(image, 1)
    for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	    shape = predictor(image, rect)
	    shape = face_utils.shape_to_np(shape)
    return shape

def read_landmarks(image):

    rects = cascade.detectMultiScale(image)

    x,y,w,h = rects[0].astype(int)
    rect = dlib.rectangle(x, y, x + w, y + h)

    face_points = predictor(image,rect).parts()
    #face_points = face_utils.shape_to_np(face_points)
    #return face_points
    landmarks = []
    for n in face_points:
        landmarks.append([n.x, n.y])
    return landmarks

def imageAlignment(inPoints, outPoints) :
    s60 = math.sin(60*math.pi/180);
    c60 = math.cos(60*math.pi/180);  
  
    inPts = np.copy(inPoints).tolist();
    outPts = np.copy(outPoints).tolist();
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
    
    inPts.append([np.int(xin), np.int(yin)]);
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
    
    outPts.append([np.int(xout), np.int(yout)]);
    
    tform = cv2.estimateRigidTransform(np.array([inPts]), np.array([outPts]), False);
    
    return tform

def gettingpointsOffset(image):
    fp = readLandmarks(image)
    facepoints = []
    n = 0
    while n<68:
        x = fp[n,0]
        y = fp[n,1]
        facepoints.append((int(x),int(y)))
        n += 1

    image1 = cv2.imread(image)
    #image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image1 = np.float32(image1)/255.0
    w= 250
    h = 250
    eyecornerdst = [ (np.int(0.3 * w ), np.int(h / 3)), (np.int(0.7 * w ), np.int(h / 3)) ]
    eyecornersrc = [facepoints[36], facepoints[45]]

    tform = imageAlignment(eyecornersrc, eyecornerdst)    
    boundaryPts = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), ( w-1, h-1 ), ( w/2, h-1 ), (0, h-1), (0,h/2) ])

    afterImage = cv2.warpAffine(image1, tform, (w,h))

    points1 = np.reshape(np.array(facepoints), (68,1, 2))

    points2 = cv2.transform(points1, tform)

    points2 = np.float32(np.reshape(points2, (68, 2)))

    return points2




offset_avg = gettingpointsOffset('averagefaceimage_seta.jpg')
offset_sml = gettingpointsOffset('averagefaceimage_setb.jpg')

######      Average smile button ###################
Smile_offset = offset_sml - offset_avg
Smile_offset = np.array(Smile_offset)

small_offset_increment =  np.zeros((68,2))
#for the slider#
small_offset_increment[48:68] = [-6, -5],[-1, -1],[-1, -1],[1, -1],[1, -1],[1, -1],[5, -5],[1, 1],[-1, 1],[-1, 1],[-1, -1],[-1, -1],[-5, -4],[-1, -1],[1, -1],[1, -1],[5, -4],[1, 1],[1, 1],[-1, 1]

'''debugging and checking lists '''
#print(small_offset_increment.shape)
#print(small_offset_increment[48:68])
#print(Smile_offset)

### Square faces ###
offset_square_avg = gettingpointsOffset('./average_smile/1a.jpg')
offset_square_sml = gettingpointsOffset('./average_smile/1b.jpg')

square_face = offset_square_sml - offset_square_avg
square_face = np.array(square_face)

### round faces ###
offset_round_avg = gettingpointsOffset('./average_smile/96a.jpg')
offset_round_sml = gettingpointsOffset('./average_smile/96b.jpg')

round_face = offset_round_sml - offset_round_avg
round_face = np.array(round_face)

### long/rectangular faces ###
offset_long_avg = gettingpointsOffset('./average_smile/71a.jpg')
offset_long_sml = gettingpointsOffset('./average_smile/71b.jpg')

rectangular_face = offset_long_sml - offset_long_avg
rectangular_face = np.array(rectangular_face)


###Warping script###
'''
img = cv2.imread('./average_smile/13a.jpg',1)
im_copy = img.copy()
points_src = read_landmarks(img)
points_src = np.array(points_src)

points_dst = points_src + Smile_offset

tshape = np.array(points_dst, np.float32)
sshape = np.array(points_src, np.float32)



tps = cv2.createThinPlateSplineShapeTransformer()


sshape = sshape.reshape(1,-1,2)
tshape = tshape.reshape(1,-1,2)

matches = list()
n= 0
while n < 72:
    matches.append(cv2.DMatch(n, n, n))
    n+= 1

tps.estimateTransformation(tshape,sshape,matches)


#img = cv2.imread('1a.jpg',1)
out_img = tps.warpImage(img)

cv2.imshow('Smiling', out_img)
cv2.imshow('Original', im_copy)
cv2.waitKey(0)
cv2.destroyAllWindows
'''

###################################
#test warp script#
'''
pnt1 = [0,0]
pnt2 = [250,0]
pnt3 = [0,300]
pnt4 = [250,300]
pnt5 = [84,210]
pnt6 = [150,240]
sshape = np.array([[0,0],[250,0],[0,300],[250,300],[72,200],[160,220]],np.float32)
tshape = np.array([pnt1, pnt2, pnt3, pnt4, pnt5, pnt6],np.float32)
#tps.estimateTransformation(tshape, sshape)
'''



'''
matches.append(cv2.DMatch(0,0,0))
matches.append(cv2.DMatch(1,1,0))
matches.append(cv2.DMatch(2,2,0))
matches.append(cv2.DMatch(3,3,0))
matches.append(cv2.DMatch(4,4,0))
matches.append(cv2.DMatch(5,5,0))
'''
'''
fig, ax = plt.subplots()
ax.set_xlim([0,2000])
ax.set_ylim([0,1500])
plt.imshow(out_img)
plt.savefig("warped")
'''