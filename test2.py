import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)
fig, ax = plt.subplots(1, 1)
ax.plot(x, y)
plt.show()
