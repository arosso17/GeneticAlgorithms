import numpy as np

sizes = [6, 4, 2]

# b = [1 * np.random.rand(y, 1) - 0.5 for y in sizes[1:]]
# print(b)
# for i in range(len(b)):
#     for j in range(len(b[i])):
#         if np.random.random() > 0.9:
#             b[i][j][0] = 1 * np.random.random() - 0.5
# print(b)
b = np.random.rand(1000)
print(b.mean(), b.std())
#
# # print([2 * np.random.rand(y, x) - 1 for x, y in zip(sizes[:-1], sizes[1:])])
#
# print([[1 * np.random.random() - 0.5] for _ in range(100)])
# # print([[np.random.rand()] for _ in range(100)])
#
# print(np.linspace(250, 50, 200 // 10 + 1))
# sizes.sort()
# a = sizes
# print(a)
