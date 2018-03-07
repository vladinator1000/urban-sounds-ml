import matplotlib.pyplot as plt

from data import data

first_column = data.iloc[:, 0]
second_column = data.iloc[:, 1]

plt.scatter(first_column, second_column)
plt.show()