import numpy as np

from layer import Conv2d

x = np.array([[[2, 3], [4, 5]]])

c = Conv2d(1, 1, 2, 0, 1)

y = c.forward(x)
print(y)
