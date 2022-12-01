# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# imgR = cv2.imread('stereovisionImgs/im0.png',0)
# imgL = cv2.imread('stereovisionImgs/im1.png',0)

# stereo = cv2.StereoBM(1, 16, 15)
# disparity = stereo.compute(imgL, imgR)

# plt.imshow(disparity,'gray')
# plt.show()

import numpy as np
import cv2
from matplotlib import pyplot as plt
imgL = cv2.imread('stereovisionImgs/im0.png',0)
imgR = cv2.imread('stereovisionImgs/im1.png',0)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()