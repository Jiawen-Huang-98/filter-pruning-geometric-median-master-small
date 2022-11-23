import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

W = 1
H = 0.08

plt.figure(figsize=(6, 4), dpi=100)
# plt.title("matplot demo")
ax = plt.gca()
ax.set_xlim(0, W)
ax.set_xlabel("Score of filters")
ax.set_ylim(0, H)
ax.set_ylabel("Frequency")

ax.add_patch(plt.Rectangle((0.1, 0.01), 0.4, 0.9, color="gray", linestyle="--", fill=False, linewidth=1))
ax.annotate('Removed by SFP', xy=(0.5, 0.6), xytext=(0.6, 0.8),
 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3"))


X = np.arange(0, 1.2, 0.05)
X_SFP =np.append(X, 0.5001)
X_SFP.sort()
X1 = np.linspace(1, W, 200)
Y1 = np.array([0.2, 0.25, 0.3, 0.15, 0.45, 0.35, 0.25, 0.5, 0.6, 0.9, 0.8, 1, 0.95, 0.8, 0.6, 0.4, 0.65, 0.5, 0.8, 0.5, 0.6, 0.5, 0.35, 0.5])
Y1 = Y1/Y1.sum()
Y_SFP = np.array([0.2, 0.25, 0.3, 0.15, 0.45, 0.35, 0.25, 0.5, 0.6, 0.9, 0.8, 0.8, 1, 0.95, 0.8, 0.6, 0.4, 0.65, 0.5, 0.8, 0.5, 0.6, 0.5, 0.7, 0.5])
Y_SFP_ave =Y_SFP/Y_SFP.sum()
Y_SFP = np.append(Y_SFP_ave[:10],Y_SFP[10:])
Yr = np.append(Y1[:10]*0.7, Y1[10:])
Yo = np.append(Y1[:10]*0.5, Y1[10:])
Yb = np.append(Y1[:10]*0.3, Y1[10:])
Yg = np.append(Y_SFP[:11]*0, Y1[10:])
Yh = Y1*0.2

#print("merge:",  )

plt.plot(X, Y1, color="purple", label="line1", linestyle="-")
plt.plot(X, Yr, color="red", label="line2", alpha=0.5)
plt.plot(X, Yo, color="orange", label="line2", alpha=0.5)
plt.plot(X, Yb, color="blue", label="line2", alpha=0.5)
#plt.plot(X, Yh, color="gray", label="line2", alpha=0.5)
plt.plot(X_SFP, Yg, color="green", label="line2", )

plt.legend()
# plt.grid(True)
# plt.grid(linestyle='--')
plt.show()
