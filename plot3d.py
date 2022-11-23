import math
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

#3D Plotting
fig = plt.figure()
ax = plt.axes(projection="3d")

X = np.arange(0,6)
Y = np.arange(1,9)
Z = np.linspace(0,1,6)

x = 0
y = 0
bottom = 0
width = 1
depth = 1
top = 0.8
use_shape=False
use_color = ['red', 'green', 'purple', '#fff200', 'blue', '#3d3d3d', 'gray', 'orange', 'pink', 'cyan']
top_rd = np.random.rand(6,9)
for x in X:
    np.random.seed(x)
    pruned = np.random.randint(1, 8, size=2)
    for y in Y:
        t = top + 0.2 * math.fabs(top_rd[x,y])

        if y < 3:
            t = 0.5 - 0.05 * x
            t1 =  (top + 0.2 * math.fabs(top_rd[x,y])) - (1/(math.exp(-x+4.5) + 1))
            t1 *= 0.2
            ax.bar3d(x, y, bottom, width*0.1, depth*0.3, t1, shade=use_shape, color=use_color[y], edgecolor="#333333")
            ax.bar3d(x, y, t1, width*0.1, depth*0.3, t, shade=use_shape, color=use_color[y], edgecolor="#333333", linestyle="--",alpha=0)
        else:
            ax.bar3d(x, y, bottom, width*0.1, depth*0.3, t, shade=use_shape, color=use_color[y], edgecolor="#333333")


# plt.plot(X, make_interp_spline(X1, Y1)(X), color="green", label="${d_r}$=0.8")


#Labeling
ax.set_xlabel('Epochs',fontsize = 25, rotation = 45, font = Path('font/Times New Roman.ttf'))
ax.set_xticks(ticks=X,labels=[95,96,97,98,99,100], fontsize = 15)
ax.set_ylabel('Filter Index',fontsize = 25, font = Path('font/Times New Roman.ttf'))
ax.set_yticks(ticks=Y,labels=np.arange(1,9), fontsize = 15)
ax.set_zlabel('Score',fontsize = 25, font = Path('font/Times New Roman.ttf'))
ax.set_zticks(Z,labels=[0,0.2,0.4,0.6,0.8,1.0], fontsize = 15)
# plt.tight_layout()
# plt.legend(fontsize = 40,frameon = False)
plt.show()