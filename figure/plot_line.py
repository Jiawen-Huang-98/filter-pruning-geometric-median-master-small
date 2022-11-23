from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("bmh")
plt.title("", fontsize=23)

X = np.array([1,2,3,4,5,6,7,8,9])
Y = np.array([93.6,93.48,93.32,93.14,92.99,92.03,93.12,92.87,93.2])
Y1 = np.ones(9) * Y[0] #np.array([93.6,93.6,93.6,93.6,93.6,93.6,93.6,93.6,93.6])


ax = plt.gca()
ax.set_xlabel("Epoch", fontsize=45)
#ax.set_ylim(0, 100)
#ax.set_xticks(np.array([1,3,5,7,9]))
ax.tick_params(labelsize=28)

x_major_locator = plt.MultipleLocator(2)
y_major_locator = plt.MultipleLocator(0.4)

# 设置x轴的间隔
ax.xaxis.set_major_locator(x_major_locator)
# 设置y轴的间隔
ax.yaxis.set_major_locator(y_major_locator)

ax.set_ylabel("Accuracy(%)", fontsize=45)
plt.plot(X,Y,lw=3, label="Other Epochs")
plt.plot(X,Y1,lw=3, ls='--', label="Epoch=1")
plt.legend(fontsize=35)
plt.show()