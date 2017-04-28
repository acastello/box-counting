# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PIL import Image
from scipy import log,log2,floor,average
# from multiprocessing import Pool
import matplotlib.pyplot as plt

BASE=2.0

def logb(base, exp):
    return log(exp)/log(base)

def pixeles(path):
    im = Image.open(path).convert('1')
    (w,h) = im.size
    pix = im.load()
    # L = [[pix[i,j] for j in range(h)] for i in range(w)]
    return (w, h, pix)

def boxcofs(L, plot=False, ndrop=0):
    meds = [(BASE**(-1-i)) for i in range(len(L))][::-1]
    x = map(lambda x: -log(x), meds)[ndrop:]
    y = map(log, L)[ndrop:]
    m,n = stats(x,y)
    def f(x):
        return m*x + n
    L2 = [y[i] / x[i] for i in range(len(x))]
    print zip(L2,L)
        
    if plot:
        plt.rc('grid', linestyle=":", color='black', alpha=0.5)
        plt.figure(1)
        plt.subplot(111)
        plt.xlabel(u'-log δ')
        plt.ylabel(u'log N(δ)')
        plt.grid()
        plt.plot(x, y, 'ro')
        plt.plot([x[0]+0.25, x[-1]-0.25],[f(x[0]+0.25), f(x[-1]-0.25)],'b-')
        plt.show()
    return m

def lcosta(L):
    cofs = [(2**(-1-i)) for i in range(len(L))][::-1]
    L2 = []
    for i in range(len(L)):
        L2.append(L[i]*cofs[i])
    return L2

def recuadros(path):
    (w, h, pix) = pixeles(path)
    n = int( floor(logb(BASE,max(w,h))) )
    L = [set() for _ in range(n)]
    maxx = maxy = 0
    minx = w-1
    miny = h-1
                
    for i in range(w):
        for j in range(h):
            if pix[i,j] < 128:
                if i < minx: minx = i
                if i > maxx: maxx = i
                if j < miny: miny = j
                if j > maxy: maxy = j
                # L[0].add((x,y))
                for k in range(n):
                    tup = (i // (BASE ** k), j // (BASE ** k))
                    L[k].add(tup)
    # factor = max(w,h) / float( max(maxx-minx, maxy-miny) )
    # print(minx,maxx,miny,maxy)
    # print factor
    return L
            
def f(path, plot=False, ndrop=1):
    L = recuadros(path)
    return boxcofs(map(len,L), plot=True, ndrop=ndrop)
    # print cofs
    # return stats(map(lambda x: -log(x), cofs), map(log, L)) 

def stats(L, N):
    l = min(len(L), len(N))
    avgx = average(L)
    avgy = average(N)
    cov = sum([ L[i]*N[i] for i in range(l) ]) / l - avgx*avgy
    varx = sum([ (L[i] - avgx)**2 for i in range(l) ]) / l 
    m = cov/varx
    n = avgy - (cov/varx) * avgx
    # print(avgx,avgy,cov,varx)
    return (m,n)    
    # la recta de regresión será: Y = m*X + n
    


    # buenos = map(lambda e: filter(lambda x: x < 255, e), l)
    # return sum(map(len, buenos))

