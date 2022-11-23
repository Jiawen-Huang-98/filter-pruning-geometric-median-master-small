# -*- encoding:utf-8 -*-
"""
@作者：Javen-Huang
@文件名：代码测试.py
@时间：2022/4/20  9:00
@文档说明:
"""
import math

import numpy as np
import torch

a = torch.load('../best_model_test/model_best_56.pth.tar')
print(a)





# a = np.arange(9)
# a = a-4
# a = a.reshape(3,3)
# print(a)
# a[a<0]=0
# print(a)
# print(abs(a))
# a[abs(a)<2] =-2
# print(a)
# for epoch in range(8):
#     if epoch % 3 == 0 and epoch != 0 :
#         print(epoch)
#
# for epoch in range(8):
#     if epoch % 3 == 0 :
#         print(epoch)



# x = np.arange(81).reshape(3,3,3,3)
# h,w,c,a = x.shape
# x_1 = x.reshape(h,w,c,a)
# print(x,x_1)
# print(x == x_1)
# if x.all() == x_1.all():
#     print(True)
# else:
#     print(False)

# print(math.sqrt(1. / 9))
# epochs = 10
# for epoch in range(1,epochs+1):
#     a = float('%.4f' % (1* (1 - epoch / epochs)))
#     print(a)




