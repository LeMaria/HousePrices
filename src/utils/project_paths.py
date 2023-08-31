from pathlib import Path

PROJECTROOT = Path(r"E:\workspace\Coding\HousePrices")


def getProjectRoot():
    return PROJECTROOT

def getDataPath():
    return PROJECTROOT.joinpath("data")

def getTrainPath():
    return getDataPath().joinpath("train.csv")

def getTestPath():
    return getDataPath().joinpath("test.csv")

