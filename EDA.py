import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class EDA:

    def __init__(self, data):

        data.boxplot(column='FIRE_SPREAD_RATE')
        plt.title('FIRE_SPREAD_RATE')
        plt.ylabel('metres / minute')
        plt.grid(True)
        plt.show()

        data['SIZE_CLASS'].value_counts().sort_index().plot(kind='bar')
        plt.title('SIZE_CLASS')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()