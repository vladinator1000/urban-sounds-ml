import matplotlib.pyplot as plt

from data import data

plt.scatter(data.iloc[:, 0], data.iloc[:, 6])
plt.show()