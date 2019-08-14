# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:36:06 2018
@author: lele
"""
import pylab as plt
from skeleton import *
import utils
import cv2
from BFS import fun
from scipy.misc import imsave


if __name__ == '__main__':
    # 读取灰度图片，并显示
    #148 324 0
    fname=548

    oImg=cv2.imread(f'img/{fname:04d}.jpg')[:,:,:3]
    img = cv2.imread(f'img/{fname:04d}_gt.jpg', 0)  # 直接读为灰度图像
    mark=img.copy()
    img=utils.inverse(img)
    #原图
    # plt.imshow(img)
    # plt.show()

    blur=img
    for i in range(2):
        blur = cv2.medianBlur(blur,5)

    #模糊图
    # plt.imshow(blur)
    # plt.show()

    # 自适应二值化函数，需要修改的是55那个位置的数字，越小越精细，细节越好，噪点更多，最大不超过图片大小

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 2)  # 换行符号 \

    # 获取自适应二值化的细化图，并显示
    iThin = Thin(th3, array)
    # 获取简单二值化的细化图，并显示
    iTwo = Two(img)
    iThin_2 = Thin(iTwo, array)
    # 论文需要，对中心线进行处理
    sb=iThin.copy()
    sb=utils.inverse(sb)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    sb = cv2.dilate(sb, kernel)  # 膨胀
    imsave('center-line.jpg',sb)
    #中心线提取后
    plt.imshow(iThin)
    plt.show()
    fun(oImg,iThin,mark)


