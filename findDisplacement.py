import numpy as np
import cv2
import time

from tracking import CentroidTracker

from randomColor import MplColorHelper

from animate3d import AnimatedScatter

listFileRoi = [["data/2021-03-31/4096-2160/4096-2160", [ 0.45, 0.34, 0.45 + 0.065, 0.34 + 0.14]]]  #x1, y1, x2, y2 (percentage)

indexFile = 0

filename = listFileRoi[indexFile][0]
resolution = (640, 360)
roi =listFileRoi[indexFile][1]
isSave = True

colorHelper = MplColorHelper("hsv", 0, 40)

cap = cv2.VideoCapture(filename+ ".avi")
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

timeStamp = None
with open(filename + "-log.txt", "r") as f:
    timeStamp = [ float(i.strip()) for i in f.readlines()]

x1, y1, x2, y2 = int(roi[0] * width), int(roi[1] * height), int(roi[2] * width), int(roi[3] * height)

if isSave :
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    real = cv2.VideoWriter( filename + "-real.mp4", fourcc, fps, resolution)
    thresh_writer = cv2.VideoWriter( filename + "-thresh.mp4", fourcc, fps, (x2-x1, y2-y1))
    masking_writer = cv2.VideoWriter( filename + "-masking.mp4", fourcc, fps, (x2-x1, y2-y1))

tracker1 = CentroidTracker(maxDisappeared=1000, maxDistance=60)

ind = 0
while(cap.isOpened()):
    
    t0 = time.time()
    
    ret, frame = cap.read()
    if ret :
        roi_image = frame[y1:y2, x1:x2].copy()

        roi_grey = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        roi_grey = cv2.GaussianBlur(roi_grey,(5,5),0)

        ret3,thresh = cv2.threshold(roi_grey,160,255,cv2.THRESH_BINARY)
        thresh = tracker1.filterBinaryImage(thresh)

        boundingBox = tracker1.getContour(thresh)
        tracker1.update(boundingBox, float(timeStamp[ind] - timeStamp[0]))
        

        roi_image = cv2.cvtColor(roi_grey,cv2.COLOR_GRAY2BGR)
        roi_image = tracker1.drawPoint(roi_image)

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
        frame = cv2.resize(frame, resolution, interpolation = cv2.INTER_AREA)

        if isSave :
            real.write(frame)
            masking_writer.write(roi_image)
            thresh_writer.write(cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB))

        cv2.imshow('frame',frame)
        cv2.imshow('Grey',roi_grey)
        cv2.imshow('roi',roi_image)
        cv2.imshow('tresholding',thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        diff = time.time() - t0
        if diff < 1/fps:
            delay = (1/fps) - diff
            time.sleep(delay)
        ind += 1
        if ind > 150:
            break
    else:
        break

dataPosition = tracker1.getPositionTracking()

desplacement = {}
desplacementScatter = {}

desplacement2 = {}

def getLength(point1, point2):
    diffX = point1[0] - point2[0]
    diffY = point1[1] - point2[1]
    length = ((diffX**2) +  (diffY**2))**0.5
    if point1[0] - point2[0] < 0:
        return -1 * length
    return length

for key, value in dataPosition.items():
    sumX = sum([i["x"] for i in value])
    sumY = sum([i["y"] for i in value])

    if len(value) > 0:
        centroid = (sumX / len(value), sumY / len(value))

        d = [getLength(centroid, (i["x"], i["y"])) for i in value]
        std = np.std(np.array(d))
        if std > 4:
            desplacement[key] = {
                "time" : [i["time"] for i in value],
                "distance" : d,
                "centroid" : centroid
            }

        if std > 6 and std < 10:
            desplacement2[key] = {
                "time" : [i["time"] for i in value],
                "distance" : d,
                "centroid" : centroid
            }

        desplacementScatter[key] = {
                "time" : [i["time"] for i in value],
                "distance" : d,
                "centroid" : centroid
            }


# x = [p[0] for p in points] 
# y = [p[1] for p in points]
# centroid = (sum(x) / len(points), sum(y) / len(points))

# import matplotlib.pyplot as plt

# axes = []

# for key, value in desplacement.items():
#     axes.append(plt.plot(value["time"], value["distance"], linewidth=0.5, label = str(key)))
#     axes[-1][0].set_color(colorHelper.get_rgb(key % 40))

# plt.xlabel('Time (s)')
# plt.ylabel('Piksel')
# plt.savefig( filename + "-chart-all.png", dpi = 1200)

# plt.clf()


# axes = []

# for key, value in desplacement2.items():
#     axes.append(plt.plot(value["time"], value["distance"], linewidth=1, label = str(key)))
#     axes[-1][0].set_color(colorHelper.get_rgb(key % 40))

# plt.xlabel('Time (s)')
# plt.ylabel('Piksel')
# plt.legend()
# plt.savefig( filename + "-chart-filter.png", dpi = 1200)

# if isSave :
#     real.release()
#     masking_writer.release()
#     thresh_writer.release()
# cap.release()
# cv2.destroyAllWindows()





