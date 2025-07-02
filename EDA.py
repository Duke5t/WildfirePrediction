import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

class EDA:

    def __init__(self, data):

        data.boxplot(column='FIRE_SPREAD_RATE')
        plt.title('FIRE_SPREAD_RATE')
        plt.ylabel('metres / minute')
        plt.grid(True)
        plt.show()
        print(f'Max: {np.max(data['FIRE_SPREAD_RATE'])}\nMin: {np.min(data['FIRE_SPREAD_RATE'])} \nMode: {stats.mode(data['FIRE_SPREAD_RATE'])}\nVariance: {np.var(data['FIRE_SPREAD_RATE']):.4f} \nStdDev: {np.std(data['FIRE_SPREAD_RATE']):.4f}\n')

        data['FIRE_SPREAD_RATE'] = np.log1p(data['FIRE_SPREAD_RATE'])
        data.boxplot(column='FIRE_SPREAD_RATE')
        plt.title('log(FIRE_SPREAD_RATE)')
        plt.ylabel('metres / minute')
        plt.grid(True)
        plt.show()
        print(f'Max: {np.max(data['FIRE_SPREAD_RATE'])}\nMin: {np.min(data['FIRE_SPREAD_RATE'])} \nMode: {stats.mode(data['FIRE_SPREAD_RATE'])}\nVariance: {np.var(data['FIRE_SPREAD_RATE']):.4f} \nStdDev: {np.std(data['FIRE_SPREAD_RATE']):.4f}\n')



        data['SIZE_CLASS'].value_counts().sort_index().plot(kind='bar')
        plt.title('SIZE_CLASS')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()