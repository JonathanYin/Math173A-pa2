import matplotlib.pyplot as plt
import pandas as pd

# read csv file
df = pd.read_csv('HW2_data.csv', header=None, names=['x', 'y'])

# extract coordinates from DataFrame
x = df['x']
y = df['y']

# draw scatter plot
plt.scatter(x, y, c='blue', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ellipse Points')
plt.grid(True)

# show plot
plt.show()
