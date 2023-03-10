import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

# x_vals = []
# y_vals = []

index = count()


def animate(i):
    data = pd.read_csv('liveplot.csv')
    x = data.index
    y1 = data['Origin_X']
    # y2 = data['Origin_Y']

    plt.cla()

    plt.plot(x, y1, label='Value 1')
    # plt.plot(x, y2, label='Value 2')

    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate)

plt.tight_layout()
plt.show()