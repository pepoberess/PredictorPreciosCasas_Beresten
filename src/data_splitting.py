import pandas as pd
import numpy as np

def split_data(data):
    ba = data[data["lat"] < 0]
    ny = data[data["lat"] > 0]
    ba = ba.sample(frac=1, random_state=42)
    ny = ny.sample(frac=1, random_state=42)

    splitBA = int(0.8 * len(ba))
    splitNY = int(0.8 * len(ny))

    trainBA = ba[:splitBA]
    valBA = ba[splitBA:]
    trainNY = ny[:splitNY]
    valNY = ny[splitNY:]

    train = pd.concat([trainBA, trainNY])
    val = pd.concat([valBA, valNY])
    return train, val

def cross_val():
    pass