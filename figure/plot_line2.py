import matplotlib.pyplot as plt
import numpy as np

plt.style.use("bmh")
plt.title("", fontsize=23)

X = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
Y = np.array([92.93,92.88,
93.05,#0.2 setted
92.96,#0.3 seeted
93.55,
93.6,93.28,92.42,91.99,92.12, 92.18])
Y1 = np.ones(11) * Y[0]


ax = plt.gca()
ax.set_xlabel("Proportion", fontsize=45)
#ax.set_ylim(0, 100)
#ax.set_xticks(np.array([1,3,5,7,9]))
ax.tick_params(labelsize=23)

x_major_locator = plt.MultipleLocator(0.2)
y_major_locator = plt.MultipleLocator(0.4)

# 设置x轴的间隔
ax.xaxis.set_major_locator(x_major_locator)
# 设置y轴的间隔
ax.yaxis.set_major_locator(y_major_locator)

ax.set_ylabel("Accuracy", fontsize=45)
plt.plot(X,Y, lw=3, label="FPAD")
plt.plot(X,Y1, lw=3,ls='--', label="Baseline")
plt.legend(fontsize=35)
plt.show()