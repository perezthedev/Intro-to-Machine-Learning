
# Ryan Aponte
# 8000758082
# Assignment #1
# Graphing Data with Scatter Plot,
# Violin Plot, and Histogram

# Used pandas & matplotlib libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# Part 1 - Scatter Plot
# Read MNIST file data
data = pd.read_csv('MNIST_100.csv')

# make two variables
y = data.iloc[:, 0]
X = data.drop('label', axis=1)

# Visualize Data
pca = PCA(n_components=2)
pca.fit(X)
PCAX = pca.transform(X)

# Plot all 10 groups, decided to use a key to label groups
plt.scatter(PCAX[0:100, 0], PCAX[0:100, 1])  # Digit 0
plt.scatter(PCAX[100:200, 0], PCAX[100:200, 1])  # Digit 1
plt.scatter(PCAX[200:300, 0], PCAX[200:300, 1])  # Digit 2
plt.scatter(PCAX[300:400, 0], PCAX[300:400, 1])  # Digit 3
plt.scatter(PCAX[400:500, 0], PCAX[400:500, 1])  # Digit 4
plt.scatter(PCAX[500:600, 0], PCAX[500:600, 1])  # Digit 5
plt.scatter(PCAX[600:700, 0], PCAX[600:700, 1])  # Digit 6
plt.scatter(PCAX[700:800, 0], PCAX[700:800, 1])  # Digit 7
plt.scatter(PCAX[800:900, 0], PCAX[800:900, 1])  # Digit 8
plt.scatter(PCAX[900:1000, 0], PCAX[900:1000, 1])
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.title('MNIST Scatter Plot')
plt.show()

# Part 2
# Violin Plot with Columns, K, M, N
csv = pd.read_csv('housing_training.csv')
res = sns.violinplot(data=csv[['15.3', '4.98', '24']])
plt.xlabel('Column')
plt.ylabel('Given Value')
plt.title('Housing Data Violin Plot')

plt.show()

# Part 3
# Visualize a Histogram
# Show Column A of data

csv['0.00632'].hist(bins=50)
plt.xlabel('Column')
plt.ylabel('Given Value')
plt.title('Housing Data Histogram (50 Bins)')
plt.show()