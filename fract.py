# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from PIL import Image, ImageColor
from scipy import log2,average
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

class Box:
    u"""
    Clase principal para generar datos intermedios y finales, los intermedios
    se generan automática y recursivamente, quedan guardados de modo que se 
    pueden llamar las funciones finales repetidamente, y si los parámetros no 
    cambian, no se recalculan datos innecesariamente

    :param filepath: nombre de la imagen a abrir
    :param umbral: (opcional) valor del color de un pixel por debajo del cual 
                   considerarlo como opaco, por defecto es 85
    
    
    - La función :func:`dimension` devuelve la dimensión box-counting calculada 
      grafico para visualizar la distribucion de la dimensión respecto a los 
      según la recta de regresión y, si no se especifca lo contrario, muestra un
      distintos recubrimientos.
    
    - :func:`imagen` genera una imagen que representa un recubrimiento concreto 
      de la imagen original
    
    Ejemplo::
        
        # nos vale el umbral estándar asique no lo especificamos
        b = Box("sierpinski.png")
        
        b.dimension(plot=False)
        # el resultado no nos convence, queremos que nos muestre el grafico
        
        b.dimension()
        # tras llamar `dimension`, se nos muestra un grafico 
        # de la distribucion de la dimensión box-counting, y no es muy lineal

        # queremos ignorar el último dato (el correspondiente al recubrimiento
        # más fino) y los 2 primeros
        b.dimension(pre=2, post=1)
        
        # podemos utilizar recubrimientos cuadrados de la longitud (en pixeles)
        # que queramos
        b.dimension(g=[1,3,9,27])
        
        # tambien podemos generar exponencialmente mas coeficientes dandole 
        # valores enteros a la `g`
        # g = 0 -> [1,2,4,8, ..]
        # g = 1 -> [1,2,3,4,6,8,12, ..]
        # g = 2 -> [1,2,3,4,5,6,7,8,10,12,14,16,20, ..]
        
        # tambien podemos excluir los puntos que vemos en el grafico dandole 
        # un valor decimal a `pre` o `post`
        
        # vemos que los valores con -log(𝛿) menores que 3.0 no nos valen
        b.dimension(pre=3.0)
        
        # generamos una imagen que ilustre el recubrimiento con lado 17 
        b.imagen(17)
        
        # generamos la misma imagen, pero sin malla, y sin superponerla sobre
        # la imagen original
        b.imagen(17, imposed=False, grid=False)
        
        # generamos la misma imagen pero especificando rojo como color para las
        # celdas que recubren y negro para la malla
        b.imagen(17, color=(255,0,0), linecolor="black")
        
        # también se puede variar la opacidad de las celdas
        b.imagen(17, alpha=200)
                    
        # en todos estos casos se ha generado y guardado la imagen 
        # sierpinski_17.png
        
    """
    
    coef = 1    
    recu = 2
    dime = 3
    
    
    def __init__(self, filepath, umbral=85):
        self.filepath = filepath
        self.umbral = umbral
        self.img = im = Image.open(filepath)
        self.w, self.h = im.size
        self.min = min(self.w, self.h)
        self.max = max(self.w, self.h)
        self.pixmap = im.convert('L').load()
        self.stage = 0
        self.g = 0
            
    def coeficientes(self, g=0, f = lambda x: x):
        if type(g) == list:
            self.coefs = self.g = g
            return g
        factors = [ 2 * i + 1 for i in range(2 ** g) ]
        pots = lambda e: [e * (2 ** i) for i in range( int(log2(self.min/e)) )]
        sides = reduce(lambda x, y: x + pots(y), factors, [])
        sides.sort()
        sides = f(sides)
        self.coefs = sides
        self.g = g
        self.stage = Box.coef
        return sides

    def recubrimientos(self, g=0):
        if self.stage >= Box.recu and g == self.g and self.coefs == self.sets.keys:
            return self.sets
        elif self.stage < Box.recu or g != self.g:
            self.coeficientes(g)
            self.sets = {}
        keys = self.sets.keys()
        L = {k: set() for k in self.coefs if not k in keys}
        
        for i in range(self.w):
            for j in range(self.h):
                if self.pixmap[i,j] < self.umbral:
                    for e,s in L.items():
                        s.add( (i//e, j//e) )
        
        self.sets = L
        self.stage = Box.recu
        return L

    def dimension(self, g=0, plot=True, pre=0, post=0):
        if self.stage < Box.dime or g != self.g:
            self.recubrimientos(g)
        L = []
        offset = log2(self.max)
        for k,v in self.sets.items():
            L.append( ( -log2(k)+offset, log2(len(v)) ) )
        L.sort()
        if type(pre) == int:
            L = L[pre:]
        elif type(pre) == float:
            L = [(i,j) for (i,j) in L if i >= pre]
            
        if type(post) == int:
            L = L[:len(L)-post]
        elif type(post) == float:
            L = [(i,j) for (i,j) in L if i <= post]

        self.dims = L
        self.m, self.n = regresion(L)
        if plot:
            self.plot()
        self.stage = Box.dime
        return L,self.m
        
    def plot(self):
        [x,y] = zip(*self.dims)
        def f(x):
            return self.m * x + self.n
        plt.rc('grid', linestyle=":", color='black', alpha=0.5)
        # plt.rc('font', family='Arial')
#         plt.rc('text', usetex=True)
        plt.figure(1)
        plt.subplot(111)
        # δ
        fp = FontProperties('Consolas')
        plt.xlabel(u'-log δ', fontproperties=fp)
        plt.ylabel(u"-log N(δ)", fontproperties = fp)
        # plt.xlabel(r'-log $\delta$')
        # plt.ylabel(r'log N($\delta$)')
        plt.grid()
        plt.plot(x, y, 'ro')
        plt.plot([x[0]-0.5, x[-1]+0.5],[f(x[0]-0.5), f(x[-1]+0.5)],'b-')
        plt.show()
    
    def imagen(self, escala, imposed=True, grid=True, color=(0x7b, 0xfd, 0xff, 0xff), 
               linecolor=(0x36, 0x44, 255, 255), alpha=127, **opts):
        self.recubrimientos(**opts)
        if self.sets.has_key(escala):
            points = self.sets[escala]
        else:
            self.coefs.append(escala)
            self.recubrimientos()
            points = self.sets[escala]
        
        if type(color) == str:
            color = ImageColor.getcolor(color, 'RGBA')
            
        if type(linecolor) == str:
            linecolor = ImageColor.getcolor(linecolor, 'RGBA')
        
        Im = Image.new('RGBA', (self.w, self.h), (255,255,255,0))
        pix = Im.load()
        
        Mask = Image.new('L', (self.w, self.h), 0)
        maskpix = Mask.load()
        for (x,y) in points:
            for i in range(escala):
                for j in range(escala):
                    if escala * x + i < self.w and escala * y + j < self.h:
                        pix[escala * x + i, escala * y + j] = color
                        maskpix[escala * x + i, escala * y + j] = alpha
        if grid:
            Grid = Image.new('RGBA', (self.w, self.h), 0x0)
            gridpix = Grid.load()
            for (x,y) in points:
                for i in range(escala+1):
                    xi = escala * x + i
                    x0 = escala * x
                    x1 = escala * (x + 1)
                    yi = escala * y + i
                    y0 = escala * y
                    y1 = escala * (y + 1)
                    if xi < self.w and y0 < self.h:
                        gridpix[xi, y0] = linecolor
                    if xi < self.w and y1 < self.h:
                        gridpix[xi, y1] = linecolor
                    if x0 < self.w and yi < self.h:
                        gridpix[x0, yi] = linecolor
                    if x1 < self.w and yi < self.h:
                        gridpix[x1, yi] = linecolor

        (base,_) = os.path.splitext(self.filepath)
        filename = base + '_' + str(escala) + '.png'
        
        if imposed:
            Imf = Image.composite(Im, self.img.convert('RGBA'), Mask)
        else:
            Imf = Im
        
        if grid:
            Imfinal = Image.composite(Grid, Imf, Grid)
        else:
            Imfinal = Imf
        
        Imfinal.save(filename)
        return filename

        
def regresion(L, N = None):
    """
    Toma o bien una lista de tuplas `(x,y)` o dos listas de numeros `[x]`,
    `[y]` y devuelve los coeficientes para formar la recta de regresion de
    los datos
    """
    if N == None:
        [L, N] = zip(*L)
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
    