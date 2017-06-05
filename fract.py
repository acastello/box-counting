# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PIL import Image
from scipy import log,log2,floor,average
from multiprocessing import Pool
import matplotlib.pyplot as plt

class box:
    """
    Clase contenedora de variables importantes. Tambien ayuda a generar 
    recursivamente los datos necesarios para cualquier operación
    """
    
    coef = 1    
    recu = 2
    dime = 3
    
    
    def __init__(self, filepath, threshold=128):
        self.filepath = filepath
        self.threshold = threshold
        self.img = im = Image.open(filepath).convert('L')
        self.w, self.h = im.size
        self.min = min(self.w, self.h)
        self.max = max(self.w, self.h)
        self.pixmap = im.load()
        self.stage = 0
        self.g = 0
            
    def coeficientes(self, g=0, f = lambda x: x):
        if type(g) == list:
            self.coefs = self.g = g
            return g
        factors = [ 2 * i + 1 for i in range(2 ** g) ]
        pots = lambda e: [e * (2 ** i) for i in range( int(log2(self.min/e)))]
        sides = reduce(lambda x, y: x + pots(y), factors, [])
        sides.sort()
        sides = f(sides)
        self.coefs = sides
        self.g = g
        self.stage = box.coef
        return sides

    def recubrimientos(self, g=0):
        if self.stage < box.recu or g != self.g:
            self.coeficientes(g)
        L = {k: set() for k in self.coefs}
        
        for i in range(self.w):
            for j in range(self.h):
                if self.pixmap[i,j] < self.threshold:
                    for e,s in L.items():
                        s.add( (i//e, j//e) )
        
        self.sets = L
        self.stage = box.recu
        return L

    def dimensiones(self, g=0, plot=True, pre=0, post=0):
        if self.stage < box.dime or g != self.g:
            self.recubrimientos(g)
        L = []
        offset = log2(self.max)
        for k,v in self.sets.items():
            L.append( ( -log2(k)+offset, log2(len(v)) ) )
        L.sort()
        self.dims = L
        self.m, self.n = regresion(L)
        if plot:
            self.plot()
        self.stage = box.dime
        return L,self.m
        
    def plot(self):
        x = map(lambda (x,_): x, self.dims)
        y = map(lambda (_,y): y, self.dims)
        def f(x):
            return self.m * x + self.n
        plt.rc('grid', linestyle=":", color='black', alpha=0.5)
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        plt.figure(1)
        plt.subplot(111)
        plt.xlabel(r'-log $\delta$')
        plt.ylabel(r'log N($\delta$)')
        plt.grid()
        plt.plot(x, y, 'ro')
        plt.plot([x[0]-0.25, x[-1]+0.25],[f(x[0]-0.25), f(x[-1]+0.25)],'b-')
        plt.show()
        
        
def pixeles(path):
    """
    :param path: localizacion de la imagen a cargar (todos los formatos soportados)
    Devuelve en una tupla: ancho, alto y mapa de pixeles (accesible con [i,j]) de la imagen
    """
    im = Image.open(path).convert('L')
    (w,h) = im.size
    pix = im.load()
    # L = [[pix[i,j] for j in range(h)] for i in range(w)]
    return (w, h, pix)

def dimensiones(sets):
    L = []
    offset = log2(sets['dim'])
    for k,v in sets.items():
        if type(k) == int:
            L.append((-log2(k)+offset, log2(len(v))))
    return sorted(L)

def regresion(L, N = None):
    """
    Toma o bien una lista de tuplas `(x,y)` o dos listas de numeros `[x]`, `[y]`
    y devuelve los coeficientes para formar la recta de regresion de los datos
    """
    if N == None:
        N = map(lambda (_,y): y, L)
        L = map(lambda (x,_): x, L)
    l = min(len(L), len(N))
    avgx = average(L)
    avgy = average(N)
    cov = sum([ L[i]*N[i] for i in range(l) ]) / l - avgx*avgy
    varx = sum([ (L[i] - avgx)**2 for i in range(l) ]) / l 
    m = cov/varx
    n = avgy - (cov/varx) * avgx
    # print(avgx,avgy,cov,varx)
    # la recta de regresión será: Y = m*X + n
    return (m,n)    

def plotdims(dims):
    x = map(lambda (x,_): x, dims)
    y = map(lambda (_,y): y, dims)
    m,n = regresion(x,y)
    def f(x):
        return m*x + n
    plt.rc('grid', linestyle=":", color='black', alpha=0.5)
    plt.figure(1)
    plt.subplot(111)
    plt.xlabel(u'-log δ')
    plt.ylabel(u'log N(δ)')
    plt.grid()
    plt.plot(x, y, 'ro')
    plt.plot([x[0]-0.25, x[-1]+0.25],[f(x[0]-0.25), f(x[-1]+0.25)],'b-')
    plt.show()
    
def final(fp, g=0):
    w,h,pix = pixeles(fp)
    coefs = coeficientes(w,h,g)
    recs = recubrimientos(w,h,coefs,pix)
    dims = dimensiones(recs)
    plotdims(dims)
    m, n = regresion(dims)
    return m
    
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
            
def f(path, ndrop=1, plot=True):
    L = recuadros(path)
    return boxcofs(map(len,L), plot, ndrop=ndrop)
    # print cofs
    # return stats(map(lambda x: -log(x), cofs), map(log, L)) 
    
# if __name__ == '__main__':
    
