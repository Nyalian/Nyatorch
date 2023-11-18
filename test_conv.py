import numpy as np

from layer import Conv2d, Flatten

x = np.array([[[1.09824655, -1.47418237],
               [1.70464524, -1.16688851]]])

c = Flatten()

y = c.forward(x)
z=c.backward(y)
print(z)
