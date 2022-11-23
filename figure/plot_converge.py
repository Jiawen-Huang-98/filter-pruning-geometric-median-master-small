import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

W = 1000
H = 1

plt.figure(figsize=(4, 3), dpi=100)
# plt.title("matplot demo")
ax = plt.gca()
ax.set_xlim(0, W)
ax.set_xlabel("Epoch",fontsize = 40)
ax.set_ylim(0, H)
ax.set_ylabel("Density",fontsize = 40)



X = np.linspace(0, W, 500)
X_tar = np.linspace(0, W, 200)

X1 = np.array([0,  998, 999, 1000])
Y1 = np.array([1,  0.2, 0.2, 0.2])

X2 = np.array([0, 598, 599, 600, 700])
Y2 = np.array([1, 0.2, 0.2, 0.2, 0.2])

X3 = np.array([0,  300, 320, 350, 400, 500, 600, 700])
Y3 = np.array([1,  0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

# Y4 = np.array([1, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])

X4 = np.array([0,  200, 210, 220, 230, 240, 250, 300, 500])
Y4 = np.array([1,  0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])

X5 = np.array([0, 50, 100, 200])
Y5 = np.array([0.2, 0.2, 0.2, 0.2])

plt.plot(X, make_interp_spline(X1, Y1)(X), color="green", label="${d_r}$=0.8")
plt.plot(X, make_interp_spline(X2, Y2)(X), color="red", label="${d_r}$=0.5")
# plt.plot(X, make_interp_spline(X1, Y1)(X), color="green", label="decay rate=0.5")
plt.plot(X, make_interp_spline(X3, Y3)(X), color="purple", label="${d_r}$=0.1")
plt.plot(X, make_interp_spline(X4, Y4)(X), color="orange", label="${d_r}$=0.01")
plt.plot(X_tar, make_interp_spline(X5, Y5)(X_tar), color="black", linestyle = "--",dashes=(10, 10), label="Target")
# 控制x和y轴的字体大小
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
# 控制图例的字体大小
ax.axes.xaxis.set_ticks([])
plt.legend(fontsize = 40,frameon = False)
plt.grid(True)
plt.grid(linestyle='--')
plt.show()
