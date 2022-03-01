import cv2
from imutils import contours
from imutils.video import FPS
from skimage import measure
import numpy as np
import time
import os
import datetime

fileName   = "4096-2160"
resolution = (4096, 2160)
duration   = 20 #second

dirPath = "data/" + str(datetime.datetime.now().date())
filePath = dirPath + "/" + fileName

try:
    os.mkdir("data")
except OSError as error:
    print(error)   

try:
    os.mkdir(dirPath)
except OSError as error:
    print(error)   

try:
    os.mkdir(filePath)
except OSError as error:
    print(error)   


indexCam = 1
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
cap = cv2.VideoCapture()
cap.open(indexCam + cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

time.sleep(2)


# start the FPS timer
fps = FPS().start()
out = cv2.VideoWriter( filePath + "/" +fileName + '.avi', fourcc, 8.0, resolution)

t0 = time.time()

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame,0)
    # write the flipped frame
    out.write(frame)
    
    f = open(filePath + "/" +fileName + '-log.txt', "a")
    f.write("{0}\n".format(time.time()))
    f.close()

    fps.update()

    if time.time() - t0 > duration:
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


f = open(filePath + "/" +fileName + '-info.txt', "a")
f.write(f"""
Resolution : {resolution[0]}, {resolution[1]}
Elasped time: {"{:.2f}".format(fps.elapsed())}
Approx. FPS: {"{:.2f}".format(fps.fps())}
""")
f.close()

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()