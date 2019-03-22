import numpy as np
import cv2
import utils
import pylab as plt
from math import sqrt
# from sympy import EmptySet
# from sympy.geometry import Line,Segment,Circle,Point,Polygon

eps=1e-8


def display_points(canvas,pts):
    for pt in pts:
        cv2.circle(canvas,utils.tuple_int(pt),3,(255,0,0))

#https://blog.csdn.net/syz201558503103/article/details/78400858
def is_circle_intersect_segment(circle, segment):  # 点p1和p2都不在圆内
    p1, p2 = segment
    O = circle.pt
    r = circle.r
    d1=p1.distTo(O)
    d2=p2.distTo(O)
    if (d1<r) ^ (d2<r):     #一点圆内一点圆外, 一定能相交
        return 1
    elif d1<r and d2<r: #两点圆内, 一定不相交
        return -1
    #开始两点圆外的判断
    if (p1.x == p2.x):
        a = 1
        b = 0
        c = -p1.x  # 特殊情况判断，分母不能为零
    elif (p1.y == p2.y):
        a = 0
        b = 1
        c = -p1.y  # 特殊情况判断，分母不能为零
    else:
        a = p1.y - p2.y
        b = p2.x - p1.x
        c = p1.x * p2.y - p1.y * p2.x

    dist1 = a * O.x + b * O.y + c
    dist1 *= dist1
    dist2 = (a * a + b * b) * r * r
    if (dist1 > dist2):  # 点到直线距离大于半径r
        return 0
    angle1 = (O.x - p1.x) * (p2.x - p1.x) + (O.y - p1.y) * (p2.y - p1.y)
    angle2 = (O.x - p2.x) * (p1.x - p2.x) + (O.y - p2.y) * (p1.y - p2.y)
    if (angle1 > 0 and angle2 > 0):  # 余弦都为正，则是锐角
        return 2
    return 0

def eq(a,b):
    if abs(a-b)<eps:
        return True
    else:
        return False

class Point(object):
    x=y=0
    def midPoint(self, obj):
        return Point((self.x+obj.x)/2,(self.y+obj.y)/2)
    def isInLine(self,line):
        a,b=line
        p=self
        def onAxis(i):
            if eq(a[i],p[i]) and eq(p[i],b[i]):
                return True
            return False
        if onAxis(0) or onAxis(1):
            return True
        e1=(p - a).norm()
        e2=(b - p).norm()
        if e1 == e2 or e1 == e2**-1:
            return True
        return False
    def isInSegment(self,seg):
        a,b=seg
        p=self
        e1=(p - a).norm()
        e2=(b - p).norm()

        def is_acute(e):
            if e.x>=0 and e.y>=0:
                return True
            return False
        if e1 == e2 :
            return True
        return False
    def project(self,l):
        '''
        当前点到另外两点的投影
        :param l: Line object
        :return: project point
        '''
        dist=self.distTo(l)
        n=l.normV()
        p1= self+n**dist
        p2= self-n**dist
        a,b=l
        if p1.isInLine(l):
            return p1
        else:
            return p2
    def distTo(self,obj):
        if isinstance(obj,Line):
            A,B,C=obj.getABC()
            x,y=self
            return abs(A*x+B*y+C)/sqrt(A**2+B**2)
        elif isinstance(obj,Point):
            return (obj-self).dist()
    def __init__(self,*arg):
        '''
        一个参数: 传入元组
        两个参数: 传入x, y
        :param arg:
        '''
        if len(arg)==1:#元组解析，包含了构造函数的情况
            self.x,self.y=arg[0]
        elif len(arg)==2:
            self.x, self.y = arg
        elif  len(arg)==0:
            self.x, self.y = 0,0
    def __eq__(self, other):
        def judge(self,i):
            if abs(self[i]-other[i])<eps:
                return True
            return False
        if judge(self,0) and judge(self,1):
            return True
        return False
    def __hash__(self):
        return hash(self.__str__())
    def __pow__(self, power, modulo=None):
        re= self.x*power,self.y*power
        return Point(re)
    def __sub__(self, other):
        '''
        向量减
        :param other:
        :return:
        '''
        re= self.x-other.x,self.y-other.y
        return Point(re)
    def __truediv__(self, other):
        re= self.x/other,self.y/other
        return Point(re)
    def __mul__(self, other):
        '''
        向量点乘
        :param other:
        :return:
        '''

        if type(other)==type(Point()):
            re= self.x * other.x+ self.y * other.y
        else:
            re= self.x * other+ self.y * other
        return re
    def __add__(self, other):
        if type(other)==type(Point()):
            re= self.x + other.x, self.y + other.y
        else:
            re= self.x + other, self.y + other
        return Point(re)
    def __xor__(self, other):
        '''
        向量叉乘
        :param other:
        :return:
        '''
        re= self.x * other.y - self.y * other.x
        return Point(re)
    def __str__(self):
        return f'({self.x},{self.y})'
    def __getitem__(self, item):
        if item==0:
            return self.x
        else:
            return self.y
    def __len__(self):
        return 2
    def __iter__(self):
        yield self.x
        yield self.y
    def __repr__(self):
        return str(self)
    def dist(self):
        return sqrt(self.x**2+self.y**2)
    def normV(self):
        return Point(self.y,-self.x)/self.dist()
    def norm(self):
        if self==Point(0,0):   #有bug
            return self
        return self/self.dist()

class Line(object):
    p1 = p2 = (0, 0)
    def __init__(self, p1=(0,0), p2=(0,0 )):
        self.p1=Point(p1)
        self.p2=Point(p2)
    def __str__(self):
        return f'[{str(self.p1)},{self.p2}]'
    def __getitem__(self, item):
        if item==0:
            return self.p1
        else:
            return self.p2
    def __len__(self):
        return 2
    def __iter__(self):
        yield self.p1
        yield self.p2
    def __repr__(self):
        return str(self)
    def getABC(self):
        x1, y1 = self.p1
        x2, y2 = self.p2
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        return A, B, C
    def intersect(self,obj):
        '''
        state: -1 两点圆内, 0 两点圆外(不相交), 1 一点圆内, 2 两点圆外(相交)
        :param obj:
        :return:
        '''
        if type(obj)==type(Circle()):   #直线与圆相交
            itn=is_circle_intersect_segment(obj,self)
            re={}
            re['state'] = itn
            if itn<=0:
                return re
            pr=obj.pt.project(self)   #投影点
            n=self.norm()
            d=obj.pt.distTo(self)
            exd=sqrt(obj.r**2-d**2)
            i1=pr+n**exd
            i2=pr-n**exd
            re['line']=self
            # return i1,i2
            if itn==2:  #两个交点
                re['intersect']=(i1,i2)
            else:       #一个交点
                if i1.isInSegment(self):
                    re['intersect'] =(i1,)
                else:
                    re['intersect'] =(i2,)
            return re
        elif isinstance(obj,Line):
            a1,b1,c1=self.getABC()
            a2,b2,c2=obj.getABC()
            x = (c1 * b2 - c2 * b1) / (a2 * b1 - a1 * b2)
            y = (a2 * c1 - a1 * c2) / (a1 * b2 - a2 * b1)
            return Point(x,y)
            # ret = self[0]
            # try:
            #     t = ((self[0][0] - obj[0][0]) * (obj[0][1] - obj[1][1]) - (self[0][1] - obj[0][1]) * (obj[0][0] - obj[1][0])) \
            #         / ((self[0][0] - self[1][0]) * (obj[0][1] - obj[1][1]) - (self[0][1] - self[1][1]) * (
            #                 obj[0][0] - obj[1][0]))
            # except BaseException:
            #     return None
            # ret.x += (self[1][0] - self[0][0]) * t
            # ret.y += (self[1][1] - self[0][1]) * t
            # return ret
    def display(self,canvas,color=(0,255,255),width=5):
        cv2.line(canvas, utils.tuple_int(self.p1),  utils.tuple_int(self.p2) , color,width)

    def substractWith(self,obj,*arg):
        '''
        与对象元素做差集. 如果self在obj内, 删除. 如果self在obj外, 删除
        :param obj:
        :return:
        '''
        if isinstance(obj,Circle):
            if len(arg)>0:  #认为直接把交点作为参数传入
                pass
    def normV(self):
        return (self.p2-self.p1).normV()
    def norm(self):
        return (self.p2-self.p1).norm()
    def dist(self):
        return sqrt((self.p1.x-self.p2.x)**2+(self.p1.y-self.p2.y)**2)
    def isNeighborWith(self,other):
        for i in self:
            for j in other:
                if i==j:
                    return True
        return False
class Circle(object):
    pt=(0,0);r=1
    def __init__(self,*args):
        if len(args)==0:
            self.pt=Point(0,0)
            self.r=1
        elif len(args)==2:  #圆心+半径
            if isinstance(args[1],int) or isinstance(args[1],float) :
                self.pt=Point(args[0])
                self.r=args[1]
            else:           #两点
                self.pt=args[0].midPoint(args[1])
                self.r=(args[1]-args[0]).dist()
        elif len(args)==3: #三点
            pass
    def pointPosition(self,point):
        '''
        点在圆内: -1
        点在圆上: 0
        点在圆外: 1
        :param point:
        :return:
        '''
        d=(point-self.pt).dist()
        if eq(d,self.r):
            return 0
        elif d<self.r:
            return -1
        else:
            return 1

class LineSet(object):
    pts=[]
    lines=[]
    def __init__(self,*args):
        if len(args)==0:
            self.lines=[]
            self.pts=[]
        elif len(args)>=1:
            self.lines=[]
            self.pts=[]
            pre=None
            for i,pt in enumerate(args):
                self.pts.append(Point(pt))
                if i:
                    self.lines.append(Line(pre,pt))
                pre=pt

        # generator polygon

    def split(self):
        if len(self)<=1:
            return []
        ans=[]
        pl=LineSet()
        pre=self[0]
        pl.addLine(pre)
        for i in range(1,len(self)):
            l=self[i]  #获取当前直线
            if pre.isNeighborWith(l):
                pl.addLine(l)
            else:       #第一个分开的点
                ans.append(pl)
                pl=LineSet()
            pre=l
        ans.append(pl)
        return ans

    def test(self,r):
        # r=50
        pt=self.lines[0].p1
        C=Circle(pt,r)
        ans=[]
        for l in self:
            it = l.intersect(C)['state']
            if it >0 :
                it = l.intersect(C)['intersect']
                for i in it:
                    ans.append(i)
        return ans

    def fun(self):
        #构造初始圆, 初始点
        d=5.
        pt = self.lines[0].p1
        C = Circle(pt, d/2.)
        pl=LineSet()
        pre=pt
        while(True):
            canvas = np.ones([500, 500, 3], np.uint8) * 255
            self.display(canvas)
            pl.display(canvas,(255,0,0))
            remove = []  #被删除点
            ans=[]  #交点
            cv2.circle(canvas, utils.tuple_int(C.pt), int(C.r), (0, 255, 0))
            for l in self:    #遍历集合,查找相交点
                dict = l.intersect(C)
                if dict['state']==-1: #删除直线
                    remove.append(l)
                elif dict['state']==1: #删除一半
                    p1=dict['intersect'][0]
                    cv2.circle(canvas, utils.tuple_int(p1), 3, (0, 0, 255))
                    ans.append(p1)
                    for i in l: #遍历当前直线的两个端点
                        if C.pointPosition(i) ==1: #这个点在圆外
                            p2=i
                    l.p1=p1
                    l.p2=p2  #绑定新的点
            # plt.imshow(canvas); plt.show() #显示
            pass
            for i in remove:
                self.lines.remove(i)
            if len(ans)>2:      #找到中心点
                break
            mid = ans[0].midPoint(ans[1])
            wid=abs((ans[1]-ans[0]).dist())
            pl.addLine(pre,mid)
            pre=mid
            d=wid*1.1
            C=Circle(mid,d/2.)
            # 输出图片
        splited=self.split()
        print(len(splited))
        splited[0].display(canvas,(255,255,0))
        splited[1].display(canvas,(0,255,255))
        plt.imshow(canvas);
        plt.show()  # 显示
        return pl

    def extractMidLine(self):
        d = 5.
        pt = self.lines[0].p1
        C = Circle(pt, d / 2.)
        pl = LineSet()
        self.recurseExtractMidLine(pl,C,pt)
        return pl

    def recurseExtractMidLine(self,pl,C,pt):
        print('len=',len(self))
        if len(self)<=2:
            return
        pre = pt
        while (True):
            # canvas = np.ones([500, 500, 3], np.uint8) * 255
            # self.display(canvas)
            # pl.display(canvas, (255, 0, 0))
            remove = []  # 被删除点
            ans = []  # 交点
            # cv2.circle(canvas, utils.tuple_int(C.pt), int(C.r), (0, 255, 0))
            for l in self:  # 遍历集合,查找相交点
                dict = l.intersect(C)
                if dict['state'] == -1:  # 删除直线
                    remove.append(l)
                elif dict['state'] == 1:  # 删除一半
                    p1 = dict['intersect'][0]
                    # cv2.circle(canvas, utils.tuple_int(p1), 3, (0, 0, 255))
                    ans.append(p1)
                    for i in l:  # 遍历当前直线的两个端点
                        if C.pointPosition(i) == 1:  # 这个点在圆外
                            p2 = i
                    l.p1 = p1
                    l.p2 = p2  # 绑定新的点
            # plt.imshow(canvas);
            # plt.show()  # 显示
            for i in remove:
                self.lines.remove(i)
            if len(ans) != 2:  # 找到中心点, 跳出循环
                print(len(ans))
                break
            if len(self)<=2:
                print('None')
                break
            mid = ans[0].midPoint(ans[1])
            wid = abs((ans[1] - ans[0]).dist())
            pl.addLine(pre, mid)
            pre = mid
            d = wid * 3
            C = Circle(mid, d / 2.)
            # 输出图片
        splited = self.split()
        # print(len(splited[0]))
        # print(len(splited[1]))
        # canvas = np.ones([500, 500, 3], np.uint8) * 255
        # self.display(canvas,(0,255,255))
        # pl.display(canvas)
        # plt.imshow(canvas);plt.show()
        for i,poly in enumerate(splited):
            #查找当前多段线对应的点
            # p=[]
            # for pt in ans:
            #     if poly.havePoint(pt):
            #         p.append(pt)
            #     if len(p)>=2:
            #         break
            # if
            if i*2+1>=len(ans):
                return
            C=Circle(ans[i*2],ans[i*2+1])  #两点做圆
            poly.recurseExtractMidLine(pl,C,pre)

        return pl

    def display(self,canvas,color=(255, 0, 0),wid=3):
        for line in self:
            cv2.line(canvas, utils.tuple_int(line.p1),  utils.tuple_int(line.p2) , color,wid)

    def addPoint(self,pt):
        if len(self.pts):
            self.lines.append(Line(self.pts[-1], pt))
        self.pts.append(pt)

    def addLine(self,*args):
        if len(args)==1:
            self.lines.append(args[0])
        elif len(args)==2:
            p1, p2=args
            self.lines.append(Line(p1,p2))

    def havePoint(self,point):
        for l in self:
            if point==l[0] or point==l[1]:
                return True
        return False
    def __len__(self):
        return len(self.lines)
    def __iter__(self):
        for l in self.lines:
            yield l
    def __getitem__(self, item):
        # if item<0:
        #     return self.lines[self]
        return self.lines[item]

    def __str__(self):
        ans='{'
        for pt in self.pts:
            ans+=str(pt)
        ans+='}'
        return ans
    def __repr__(self):
        return str(self)

class PointSet(object):
    def __init__(self,*args):
        # if len(args)==0:
        #     self.pts=[]
        # else:
        self.pts=list(Point(x) for x in args)
        self.pts.sort(key=lambda x: x[0])
        self.X=None
    def __str__(self):
        re="{"
        for pt in self.pts:
            re+=pt.__str__()
        return re+"}"
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.pts)
    def __iter__(self):
        for l in self.pts:
            yield l
    def __getitem__(self, item):
        # if item<0:
        #     return self.lines[self]
        return self.pts[item]
    def getXY(self):
        X,Y=[],[]
        for pt in self:
            X.append(pt.x)
            Y.append(pt.y)
        return X,Y
    def display(self,pattern='plot',*arg,**kwargs):
        X,Y=self.getXY()
        exec(f"plt.{pattern}(X,Y,*arg,**kwargs)")

    def buildMidPointsLine(self):
        re=PointSet()
        pre=self[0]
        for i in range(1,len(self)):
            pt=self[i]
            re.pts.append(pre.midPoint(pt))
            pre=pt
        return re
    def fitToLineSet(self, other, delta=3):
        '''
        len(other)>len(self)
        :param other:
        :param delta:默认为3，意思为将other点集不断拟合减小，直到len(other) - len(self) = 2
        :return:
        '''
        #开始循环
        while len(other)-len(self)>=delta:
            vis={}
            pair={}
            choose=[]
            for pt in other:
                vis[pt]=False
            #寻找距离self最近的n个点
            for p1 in self:
                min_pt = None;min_dist = 100000000
                for p2 in other:
                    dist=p1.distTo(p2)
                    if (not vis[p2]) and dist<min_dist:
                        min_dist=dist
                        min_pt=p2
                vis[min_pt] = True
                choose.append(min_pt)
                pair[min_pt]=p1
            #目标点集中，所有点向self集靠近
            for pt in choose:
                mid=pt.midPoint(pair[pt])
                pt.x=mid.x
                pt.y=mid.y
            other.pts.sort(key=lambda x: x[0])
            other=other.buildMidPointsLine()
            # other.display()
            yield other #点集数每次减少1个
        # return other
    def insertX(self,x,st=0):
        L=Point();R=Point();start=0;end=0
        for i in range(st,len(self)):
            if self[i].x<x:
                L=self[i]
                start=i
            else:
                R=self[i]
                end=i
                break
        dx=R.x-L.x
        dy=R.y-L.y
        y=L.y+dy*(dx/x)
        self.pts.insert(start+1,Point(x,y))
        return end
    def containX(self,x):
        self.X,_=self.getXY()
        return x in self.X
    def getIndexOfX(self,x):
        self.X, _ = self.getXY()
        for i,tx in enumerate(self.X):
            if tx>x:
                return i-1

