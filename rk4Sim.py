import pandas as pd

globoDataRaw = pd.read_csv("Tracker_globo_Raw_Data.txt", sep=";", skiprows=2, names=["t", "y", "v", "a", "_"])
