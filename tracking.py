import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
from skimage import measure
import cv2

from randomColor import MplColorHelper

class CentroidTracker:
    def __init__(self, maxDisappeared=1, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.positionTracking = OrderedDict()

        self.width = 0
        self.height = 0

        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

        self.colorHelper = MplColorHelper("hsv", 0, 40)

    def changeSize(self, w, h):
        self.width = w
        self.height = h

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.positionTracking[self.nextObjectID] = []
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def getPositionTracking(self):
        return self.positionTracking.copy()

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        
    def clear(self):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.positionTracking = OrderedDict()
        self.disappeared = OrderedDict()

    def getCurrentID(self):
        return self.nextObjectID

    def getPosition(self):
        return self.objects.copy()

    def manualUpdate(self, x, y):
        if len(self.objects) > 0:
            self.objects[0][0] = int(x * self.width)
            self.objects[0][1] = int(y * self.height)

    def update(self, point, t):
        #point = [[cx,cy,dx,dy]]
        if len(point) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(point), 4), dtype="float")
        
        for i, (cX, cY,dx,dy) in enumerate(point):
            inputCentroids[i] = (cX, cY,dx,dy)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            
            D = dist.cdist(inputCentroids[:,0:2],np.array(objectCentroids)[:,0:2])
            
            same_obj = {}
            for n ,val in enumerate(D):
                ind = val.argmin()
                same_obj[n] = None
                # if val[ind]< (inputCentroids[n][2]//2):
                if val[ind] < self.maxDistance:
                    same_obj[n] = ind
                    
            for key in same_obj:
                if same_obj[key] is not None:
                    onb_active = objectIDs[same_obj[key]]
                    self.objects[onb_active] = inputCentroids[key]

                    self.positionTracking[onb_active].append({
                        "time" : t,
                        "x" :  inputCentroids[key][0],
                        "y" :  inputCentroids[key][1]
                    })

                    self.disappeared[onb_active] = 0
                else:
                    self.register(inputCentroids[key])

    def filterBinaryImage(self, thresh):
        thresh = cv2.dilate(thresh, None, iterations=1)
        thresh = cv2.erode(thresh, None, iterations=1)

        # # perform a connected component analysis on the thresholded
        # # image, then initialize a mask to store only the "large"
        # # components
        # labels = measure.label(thresh, connectivity=2, background=0)
        # mask = np.zeros(thresh.shape, dtype="uint8")
        # # loop over the unique components
        # for label in np.unique(labels):
        #     # if this is the background label, ignore it
        #     if label == 0:
        #         continue
        #     # otherwise, construct the label mask and count the
        #     # number of pixels 
        #     labelMask = np.zeros(thresh.shape, dtype="uint8")
        #     labelMask[labels == label] = 255
        #     numPixels = cv2.countNonZero(labelMask)
        #     # if the number of pixels in the component is sufficiently
        #     # large, then add it to our mask of "large blobs"
        #     if numPixels > 100:
        #         mask = cv2.add(mask, labelMask)

        return thresh

    def getContour(self, thresh):
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        H,W = thresh.shape
        size = 0
        data = []
        for c in contours:
            x,y,dx,dy = cv2.boundingRect(c)  
            cx = x+(dx//2)
            cy = y+(dy//2)
            data.append([cx,cy,dx,dy])
        return data

    def drawPoint(self, frame):
        ind = 0
        for key, item in self.objects.items():
            ind += 1
            cx = int(item[0])
            cy = int(item[1])
            color = self.colorHelper.get_rgb(key % 40)
            cv2.circle(frame, (int(cx), int(cy)), int(4),(int(color[2] * 255), int(color[1] * 255), int(color[0] * 255)), -1)
        return frame