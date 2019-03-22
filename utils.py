import numpy as np
import pylab as plt
from scipy.misc import imsave

def merge_cmd(*lst):
    cmd=''
    for s in lst:
        cmd+=s+' '
    return cmd

def reverse_gray(img):
    mark = img > 50  # 获取裂缝
    shape = img.shape
    ans = np.ones(shape, np.uint8) * 255
    ans[mark] = 0
    return ans

def jpg_to_bmp(jpg,bmp):
    img=plt.imread(jpg)
    imsave(bmp,img)

def tuple_int(lst):
    ans=[]
    for x in lst:
        ans.append(int(x))
    return tuple(ans)

def list_int(lst):
    ans=[]
    for x in lst:
        ans.append(int(x))
    return ans

def inverse(img):
    logic=img>60
    ans=img.copy()
    ans[logic]=0
    ans[~logic]=255
    return ans

def Point2tuple(pt):
    return (int(pt.x),int(pt.y))

def tuple2htmlColor(tu):
    ans="#"
    for elem in tu:
        s='%02x' % elem
        ans+=s
    return ans