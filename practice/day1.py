"""
Day 1: Tensor 基础练习
规则:
  1. 每题先在 # 预测: 后面写下你的预测,再写代码运行验证。
  2. 预测错的题,记到 notes/02_pytorch_tensors.md。
  3. 自己写,不要复制粘贴。语法不确定时查官方文档。
"""
import numpy as np
import torch


# ============================================================
# A. 创建与属性
# ============================================================

# 1. 创建 t = torch.arange(24),打印 shape、dtype、ndim、numel()
t = torch.arange(24)
print(t.shape) # [24], torch.arange() returns a 1-D tensor
print(t.dtype) # int64
print(t.ndim)  # 1
print(t.numel()) # 24


# 2. 把 t 变成 shape (2, 3, 4),命名为 t3
t3 = t.reshape(2, 3, 4)


# 3. 分别创建 shape (3, 3) 的:全 0、全 1、单位矩阵、[0,1) 均匀随机、标准正态随机
t0 = torch.zeros([3,3])
t1 = torch.ones([3, 3])
t2 = torch.eye(3,3)
t4 = torch.rand(3, 3)
t5 = torch.randn(3,3)

# print(t0)
# print(t1)
# print(t2)
# print(t4)
# print(t5)


# ============================================================
# B. 索引与切片(基于 t3)
# ============================================================

# 4. 取出第 1 个"矩阵"(index 0)
first = t3[0]


# 5. 取出所有矩阵的最后一行
# 预测 shape: (2,4)
last = t3[:,-1,:]
print(last)


# 6. 取出所有大于 20 的元素(boolean mask),结果是几维?
# 预测: 1-D
result = t3[t3 > 20]
print(result.shape)


# 7. t3[:, 1, 2] 的 shape 是什么?
# 预测 shape: (2,)
# 为什么(1-2 句): 保留所有ndim0，保留dim1的index 1，保留dim 2的index 2
result7 = t3[:, 1, 2]
print(result7.shape)


# ============================================================
# C. dtype
# ============================================================

# 8. x = torch.tensor([1, 2, 3]),打印 x / 2 和 x // 2 的 dtype,解释为什么不同
# 预测: float32 和 int64,
x= torch.tensor([1, 2, 3])
y1 = x / 2
y2 = x // 2
print(y1.dtype)
print(y2.dtype)
# 解释: x/2 is true division and it always returns float number. x // 2 returns integer and it rounds down to the nearest integer.


# 9. 把 x 转成 float32,再转回 int64
x_float = x.to(torch.float32)
print(x_float.dtype)
x_back = x_float.to(torch.int64)
print(x_back.dtype)



# 10. torch.tensor([1, 2, 3]) + torch.tensor([0.5, 0.5, 0.5]) 的 dtype 是什么?
# 预测: float32
a = torch.tensor([1, 2, 3])
b = torch.tensor([0.5, 0.5, 0.5])
c = a + b
print(c.dtype)

# ============================================================
# D. 和 NumPy 互转、内存共享
# ============================================================

# 11. a = torch.zeros(3),n = a.numpy(),执行 n[0] = 5,打印 a
# 预测: 改变了a的数值
a = torch.zeros(3)
print("oringin a:", a)
n = a.numpy()
n[0] = 5
print("after updated a: ", a)
print("after updated n: ", n)

# 12. 用 a.clone().numpy() 做同样操作,a 变了吗?
# 预测: no
n1 = a.clone().numpy()
print("n1 = a.clone(): ", n1)
n1[0] = 0
print("updated n1[0] = 0: ", n1)
print("origin a: ", a)


# 13. 对比 torch.tensor(n) 与 torch.from_numpy(n):修改 n 后,哪个会跟着变?写代码证明
# 预测: from_numpy will change
n = np.array([1., 2., 3.])
a1 = torch.tensor(n)
a2 = torch.from_numpy(n)
n[0] = 100
print(n, a1, a2)


# ============================================================
# E. view 与 copy
# ============================================================

# 14. b = t.reshape(4, 6),修改 b[0, 0] = 100,打印 t[0]。这是 view 还是 copy?
# 预测: view
b = t.reshape(4, 6)
b[0, 0] = 99
print(t[0])

# 15. 对 b.T 调用 .view(24),会发生什么?(先预测,再运行,记下报错信息)
# 预测:
b1 = b.T
b1.view(24)

# 报错信息:
# 为什么(明天讲 contiguous,今天先写你的猜测):


# # ============================================================
# # F. 小挑战(做不完可跳过)
# # ============================================================

# # 16. 不用循环,创建 (5, 5) 的棋盘 tensor:相邻格子 0/1 交替
# # 提示:arange + 取模,或切片赋值 [::2, ::2]
# # TODO
