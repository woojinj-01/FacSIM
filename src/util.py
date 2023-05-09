"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""
import pandas as pd
import numpy as np
import error
import os
from enum import Enum, auto

lengthCompareThreshold = 3
hammingDistanceRatioCompareThreshold = 0.25

#iterator for each file (since column arrangement varies among files)

class FileExt(Enum):
    CSV = auto()
    XLSX = auto()

@error.callStackRoutine
def strToFileExt(argFileExtStr: str) -> FileExt:
    match argFileExtStr:
        case '.csv':
            return FileExt.CSV
        case '.xlsx':
            return FileExt.XLSX
        case _:
            error.LOGGER.report("Invalid Argument", error.LogType.WARNING)
            return None
        
@error.callStackRoutine
def fileExtToStr(argFileExt: FileExt) -> str:
    match argFileExt:
        case FileExt.CSV:
            return '.csv'
        case FileExt.XLSX:
            return '.xlsx'
        case _:
            error.LOGGER.report("Invalid Argument", error.LogType.WARNING)
            return None

class rowIterator:

    @error.callStackRoutine
    def __init__(self, argColumnNames: pd.core.indexes.base.Index, argColumnToFind: str) -> None:
        self.columnNames: pd.core.indexes.base.Index = argColumnNames
        self.index: int = 0
        self.targetColumn: str = argColumnToFind

    @error.callStackRoutine
    def __iter__(self):
        return self
    
    @error.callStackRoutine
    def __next__(self):

        while(self.targetColumn != self.columnNames[self.index].split('.')[0]):
            self.index += 1

            if(self.index>=len(self.columnNames)):
                raise StopIteration

        self.index+=1

        return self.index-1

    @error.callStackRoutine
    def resetIndex(self):
        self.index = 0
    
    @error.callStackRoutine
    def changeTargetColumnTo(self, argColumnToFind: str):
        self.targetColumn: str = argColumnToFind

    @error.callStackRoutine
    def changeTargetAndReset(self, argColumnToFind: str):
        self.changeTargetColumnTo(argColumnToFind)
        self.resetIndex()

#better way for string compare (strongly recommended)
@error.callStackRoutine
def areSameStrings(argStr1: str, argStr2: str) -> int:

    if(type(argStr1) != str):
        return 0
    
    if(type(argStr2) != str):
        return 0


    if(abs(len(argStr1) - len(argStr2) > 3)):
       return 0
    
    if(argStr1.lower() == argStr2.lower()):
        return 1
    
    compareLength = min([len(argStr1), len(argStr2)])
    hammingDistance = 0

    for index in range(compareLength):
        if(argStr1[index] != argStr2[index]):
            hammingDistance += 1


    return int((hammingDistance/compareLength) <= hammingDistanceRatioCompareThreshold)

@error.callStackRoutine
def isEmptyData(argData: str) -> int:
    return int('.' == argData)

@error.callStackRoutine
def getRankBasedOn(argScoreList: list) -> list:

    argScoreListSorted = sorted(argScoreList, reverse = False)
    scoreToRankMap = {}
    rankList = []

    prevScore = -99999

    score = 0 
    rank = 0

    for index in range(len(argScoreListSorted)):

        score = argScoreListSorted[index]

        if(prevScore < score):
            rank += 1
        
        scoreToRankMap[score] = rank

        prevScore = score


    for score in argScoreList:
        rankList.append(scoreToRankMap[score])

    return rankList 

@error.callStackRoutine
def calGiniCoeff(list):

    if(0 == len(list)):
        return None
    
    list.sort()

    totalNum = len(list)
    totalSum = sum(list)
    percentage_delta = np.float32((1 / (totalNum-1) * 100))
    height_1 = np.float32(list[0]/totalSum*100)
    height_2 = np.float32(list[0]+list[1]/totalSum*100)

    area_AnB = (100 * 100)/2
    area_B = 0

    for i in range(totalNum-1):
        area_B += np.float32(percentage_delta * (height_1 + height_2)/2)

        if(totalNum-2 != i):
            height_1 = height_2
            height_2 += np.float32(list[i+2]/totalSum*100)

    return np.float32((area_AnB - area_B)/area_AnB)
        
@error.callStackRoutine
def readFileFor(argFilePath: str, argFileExtensionReq: list):   
    #recommended when converting a file to a dataframe. supports multiple file extension to be allowed
    #returns empty dataframe on read fail

    if(not os.path.isfile(argFilePath)):
        error.LOGGER.report("Invalid File Path", error.LogType.WARNING)
        return pd.DataFrame()
    
    fileExt = strToFileExt(os.path.splitext(argFilePath)[1])
        
    if(fileExt not in argFileExtensionReq):
        error.LOGGER.report("Invalid File Extension", error.LogType.WARNING)
        return pd.DataFrame()
    
    match fileExt:
        case FileExt.CSV:
            return pd.read_csv(argFilePath)
        case FileExt.XLSX:
            return pd.read_excel(argFilePath)
        case _:
            error.LOGGER.report("File Extension Not Supported", error.LogType.WARNING)
            return pd.DataFrame()

@error.callStackRoutine
def callAndPrint(argFunction):
    def wrapper(*args, **kwargs):

        returnValue = argFunction(*args, **kwargs)
        print(returnValue)
        
        return returnValue
    return wrapper

@error.callStackRoutine
def getValuesListFromDict(argDict: dict) -> list:
    return list(argDict.values())

@error.callStackRoutine
def getKeyListFromDict(argDict: dict) -> list:
    return list(argDict.keys())

if(__name__ == '__main__'):

    error.LOGGER.report("This Module is Not for Main Function", error.LogType.CRITICAL)




