import numpy as np
import matplotlib.pyplot as plt


# sin_theta = np.sqrt(np.random.rand(1000000))
# angle = np.arcsin(sin_theta)
angle = np.arcsin(np.random.rand(100000))

plt.hist(angle, bins=100, density=True, label="Sampled from arcsin(U[0,1])")
plt.plot(np.linspace(0, np.pi/2, 1000), np.cos(np.linspace(0, np.pi/2, 1000)), label="cos(x)")
plt.legend()
plt.show()