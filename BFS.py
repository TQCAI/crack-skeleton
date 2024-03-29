from queue import Queue
import numpy as np
import pylab as plt
from ComputationalGeometry import *
from color_index import *
from graphviz import Digraph
from scipy.misc import imsave

#8连通分量
dt=[
    (-1, -1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1),
]

# 树形结点
class Node:
    def __init__(self,lines=None,pts=None):
        self.lines=lines
        self.pts=pts
        self.children=[] # 后代也是树形结点
        # # 名字
        # self.name=""
        # # 几何信息
        # self.area=0
        # self.length=0
        # self.meanWidth=0

# 颜色下标
global ix,treeIndex,ptID
ix=0
treeIndex=1
ptID=1

def isValid(x,y,img):
    bx,by=img.shape
    if 0<=x<bx and 0<=y<by:
        return True
    return False

def findStartPoint(x,y,img,vis,startTh=80):
    vq = Queue()
    vq.put((x, y))
    vl = [(x, y)]
    vis[x][y] = 1
    last = tuple()
    while not vq.empty():
        cx, cy = vq.get()
        for dx, dy in dt:
            tx, ty = cx + dx, cy + dy
            if isValid(tx, ty,img) and vis[tx][ty] == 0 and img[tx][ty] == 0:
                vq.put((tx, ty))
                vl.append((tx, ty))
                vis[tx][ty] = 1
                last = tx, ty
                break  # 最大的区别，找到一个目标然后退出，只朝着一个方向跑
    # if len(vl)<startTh:
    #
    #     return None
    # 找完了之后，清空
    for pt in vl:
        vis[pt]=0
    return last


def getLines(l,dN=3):
    '''对于得到的裂缝向量，进行抽样'''
    subL=l[::dN]
    lines=LineSet(*subL)
    return lines

def bfs(x,y,img,vis):

    q = Queue()
    q.put((x, y))
    l = [(y, x)]
    vis[x][y] = 1
    while not q.empty():
        cx, cy = q.get()
        cnt=0
        tmpL=[]
        for dx, dy in dt: # 8连通遍历
            tx, ty = cx + dx, cy + dy
            if isValid(tx, ty,img) and vis[tx][ty] == 0 and img[tx][ty] == 0:
                q.put((tx, ty))
                l.append((ty, tx))
                vis[tx][ty] = 1
                cnt+=1
                tmpL.append((tx,ty))
        if cnt>1: # 判断是否找到关节点
            lines=getLines(l)
            cur = Node(lines,l)  # 当前结点
            # print(cnt)
            while cnt: # 清理队列元素
                cnt-=1
                u=q.get()
                tx,ty=u
                vis[tx,ty]=0
                node=bfs(tx,ty,img,vis)  # 开始子节点遍历
                cur.children.append(node)
            # 对于孩子结点，进行一个判断
            for i,child in enumerate(cur.children):
                tmpLines=child.lines
                tmpPts=child.pts
                if len(tmpPts)<50: # 不符合阈值条件
                    cur.children.remove(child)
            # 如果只有一个分支，合并到父结点上
            if len(cur.children)==1:
                child=cur.children[0]
                cur.pts+=child.pts
                cur.lines.lines+=child.lines.lines
                cur.children=[]
            return cur #完成操作
    # 如果完成的是单条裂缝
    lines = getLines(l)
    cur = Node(lines,l)  # 当前结点
    return cur  # 完成操作

    # print(lines)
    # lines.display(canvas)
    # plt.imshow(canvas)
    # plt.show()

def printLength(lines):
    ans=0
    for line in lines:
        ans+=line.dist()
    print(f"{ans:.2f}",end='\t')
    return ans

def printArea(branchPT,centerPT,mark,radius=15):
    # print('branchPT',branchPT)
    if branchPT is not None:
        for x in range(branchPT[1]-radius,branchPT[1]+radius):
            for y in range(branchPT[0]-radius,branchPT[0]+radius):
                mark[x,y]=0
    # BFS算面积
    q = Queue()
    u=(centerPT[1], centerPT[0])
    q.put(u)
    vis=np.zeros_like(mark)
    vis[u] = 1
    area = 0
    while not q.empty():
        cx, cy = q.get()

        tmpL=[]
        for dx, dy in dt: # 8连通遍历
            tx, ty = cx + dx, cy + dy
            if isValid(tx, ty,mark) and vis[tx][ty] == 0 and mark[tx][ty] >60:
                q.put((tx, ty))
                vis[tx][ty] = 1
                area+=1
    print(area,end='\t')
    return area

def displayTree(root,img,mark):
    global ix, treeIndex,ptID
    g = Digraph(f'tree-{treeIndex}')
    preNode=None

    bx, by,_ = img.shape
    # canvas = cv2.flip(img, 1, dst=None)
    canvas=img
    # canvas=img.transpose((0,1,2))   #转置
    # canvas = cv2.flip(canvas, 1, dst=None)  #水平镜像
    nodeIndex=1
    # BFS 遍历
    q=Queue()
    q.put((root,''))
    while(not q.empty()):
        u=q.get()
        node=u[0]
        fatherName=u[1]
        lines = node.lines
        pts=node.pts
        name=f'{treeIndex}-{nodeIndex}'
        color=color_index[ix%len(color_index)]
        # 画裂缝
        lines.display(canvas,color=color)
        # 画标注
        N=len(lines)
        cPt=lines[N//2][0]
        cPt=utils.Point2tuple(cPt)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # tuple(map(int,self.pts[-1]))
        cv2.putText(canvas, name, cPt, font, 1, (255, 255, 255), 2)
        # 画关节点
        spt=None
        if node is not root:
            spt=lines[0][0]
            cv2.circle(canvas, utils.Point2tuple(spt), 7, (0, 0, 0), 2)
            # cv2.putText(canvas, f"pt-{ptID}", txtPt, font, 1 , (255, 255, 255), 1)
            # ptID+=1
        # 添加孩子
        for child in node.children:
            q.put((child,name))
        # 画树形图
        g.node(name=name, color=utils.tuple2htmlColor(color))
        if fatherName:
            g.edge(fatherName,name,color='black')
        ix+=1
        nodeIndex+=1
        # 展示几何信息
        print(name,end='\t')
        length=printLength(lines)
        area=printArea(spt,pts[len(pts)//2],mark)
        print(f"{area/length:.2f}")

    plt.imshow(canvas)
    plt.show()
    g.view()
    treeIndex+=1

def fun(img,crack,mark):
    print('ID\tlength\tarea\tmean width')
    crackTrees=[]
    vis=np.zeros_like(crack, dtype='uint8')
    for x,rows in enumerate(crack):
        for y,elem in enumerate(rows):
            if elem==0 and vis[x,y]==0: #发现了中心线
                #首先跑到头
                last=findStartPoint(x, y, crack, vis)
                if last is None:
                    continue
                #找到起始点后，开始正式执行dfs
                cx,cy=last
                tree=bfs(cx, cy, crack, vis)
                crackTrees.append(tree)
                displayTree(tree, img,mark)
    imsave('ans.jpg',img)
    # displayTrees(crackTrees, img)


