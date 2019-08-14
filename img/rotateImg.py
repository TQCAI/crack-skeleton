import cv2
from scipy.misc import imsave

fname = 148

oImg = cv2.imread(f'{fname:04d}.jpg')
img = cv2.imread(f'{fname:04d}_gt.jpg', 0)  # 直接读为灰度图像
h,w,_=oImg.shape
img=cv2.resize(img,(h,w))
imsave(f'{fname:04d}_gt.jpg',img)
