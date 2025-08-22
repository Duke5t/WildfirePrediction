import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

class EDA:

    def __init__(self, data):
        print("-------EDA--------")

        data.boxplot(column='FIRE_SPREAD_RATE')
        plt.title('Fire Spread Rate Distribution')
        plt.ylabel('metres / minute')
        plt.grid(True)
        plt.show()
        print(f'Max: {np.max(data['FIRE_SPREAD_RATE'])}\nMin: {np.min(data['FIRE_SPREAD_RATE'])} \nMode: {stats.mode(data['FIRE_SPREAD_RATE'])}\nVariance: {np.var(data['FIRE_SPREAD_RATE']):.4f} \nStdDev: {np.std(data['FIRE_SPREAD_RATE']):.4f}\n')

        data['FIRE_SPREAD_RATE'] = np.log1p(data['FIRE_SPREAD_RATE'])
        data.boxplot(column='FIRE_SPREAD_RATE')
        plt.title('Transformed Fire Spread Rate Distribution')
        plt.ylabel('metres / minute')
        plt.grid(True)
        plt.show()
        print(f'Max: {np.max(data['FIRE_SPREAD_RATE'])}\nMin: {np.min(data['FIRE_SPREAD_RATE'])} \nMode: {stats.mode(data['FIRE_SPREAD_RATE'])}\nVariance: {np.var(data['FIRE_SPREAD_RATE']):.4f} \nStdDev: {np.std(data['FIRE_SPREAD_RATE']):.4f}\n')


        data['SIZE_CLASS'].value_counts().sort_index().plot(kind='bar')
        plt.title('SIZE_CLASS')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()
        


        # Count values per SIZE_CLASS
        size_counts = data['SIZE_CLASS'].value_counts().sort_index()

        # Create bar chart
        ax = size_counts.plot(kind="bar")

        # Titles and labels
        plt.title("Size Class Distribution", fontsize=14, weight="bold")
        plt.ylabel("Count", fontsize=12)
        plt.xlabel("SIZE CLASS", fontsize=12)

        # Rotate x-axis labels
        plt.xticks(rotation=0, ha="right")

        # Add gridlines on y-axis only
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Add value labels on top of bars with small offset
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height() + 0.01 * size_counts.max()),
                ha="center", va="bottom", fontsize=9, color="black"
            )

        # Increase y-limit by 10% to avoid cutoff
        ax.set_ylim(0, size_counts.max() * 1.1)

        plt.tight_layout()  # avoid label cutoff
        plt.show()
        
        cols = ["FUEL_TYPE",
            "TEMPERATURE",
            "RELATIVE_HUMIDITY",
            "WEATHER_CONDITIONS_OVER_FIRE",
            "WIND_SPEED",
            "FIRE_POSITION_ON_SLOPE",
            "FIRE_START_DATE"]

        null_counts = data[cols].isnull().sum()

        # Create bar chart
        ax = null_counts.plot(
            kind="bar"
        )

        # Titles and labels
        plt.title("Null Value Frequency per Feature", fontsize=14, weight="bold")
        plt.ylabel("Count of Nulls", fontsize=12)
        plt.xlabel("Features", fontsize=12)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha="right")

        # Add gridlines on y-axis only
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(
                str(p.get_height()),
                (p.get_x() + p.get_width() / 2., p.get_height() + 0.01 * null_counts.max()),  
                ha="center", va="bottom", fontsize=9, color="black"
            )

        # Increase y-limit by 10% to avoid cutoff
        ax.set_ylim(0, null_counts.max() * 1.1)

        plt.tight_layout()  # avoid label cutoff
        plt.show()