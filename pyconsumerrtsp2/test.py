import base64
import io
import cv2 as cv21
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

# base64_string = "CQrIQA=="
# imgdata = base64.b64decode(base64_string)
# print(imgdata)
# nparr = np.fromstring(imgdata, np.uint8)

# nparr = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
# img_np = cv21.imdecode(nparr, cv21.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
# nparr = nparr.reshape((4,4,2))
# print(type(img_np))
k = np.array([[1,2],[3,4]])
l = np.array([[5,6],[7,8]])
m = np.array([[9,10],[11,12]])
nparr = np.stack((k, l, m))
nparr = np.moveaxis(nparr,0,-1)
print(nparr)
print(nparr.shape)

