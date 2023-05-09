"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""
import pandas as pd
import numpy as np
import error
import os
import math
from enum import Enum, auto

lengthCompareThreshold = 3
hammingDistanceRatioCompareThreshold = 0.25


class Gender(Enum):
    MALE = auto()
    FEMALE = auto()

@error.callStackRoutine
def strToGender(argGenderString: str) -> Gender:

    maleCandidateList = ['male', 'M']
    femaleCandidateList = ['female', 'F']
    
    if(ifStrMatchesAmong(argGenderString, maleCandidateList)):
        return Gender.MALE
    elif(ifStrMatchesAmong(argGenderString, femaleCandidateList)):
        return Gender.FEMALE
    else:
        return None
    
@error.callStackRoutine
def genderToStr(argGender: Gender) -> str:

    match argGender:
        case Gender.MALE:
            return 'M'
        case Gender.FEMALE:
            return 'F'
        case _:
            return str()

#class FileExt
#every file extension should be represented using this Enum class
class FileExt(Enum):
    CSV = auto()
    XLSX = auto()

#converts a string to a FileExt instance 
# ex) argument: '.csv', return value: FileExt.CSV
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

#converts a FileExt instance to astring
# ex) argument: FileExt.CSV, return value: '.csv'
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
        
@error.callStackRoutine
def getMean(argList: list):
    error.LOGGER.report("Careful to use. This function is not error tolerant yet", error.LogType.WARNING)

    if(0 == len(argList)):
        return 0
    
    return sum(argList)/len(argList)


#iterator for each file (since column arrangement varies among files)
class rowIterator:

    @error.callStackRoutine
    def __init__(self, argColumnNames: pd.core.indexes.base.Index, argColumnToFind: str) -> None:
        self.columnNames: pd.core.indexes.base.Index = argColumnNames
        self.index: int = 0
        self.targetColumn: str = argColumnToFind
        self.approximation = 0

    @error.callStackRoutine
    def __iter__(self):
        return self
    
    @error.callStackRoutine
    def __next__(self):

        if(0 == self.approximation):
            while(self.targetColumn != self.columnNames[self.index].split('.')[0]):
                self.index += 1

                if(self.index>=len(self.columnNames)):
                    raise StopIteration
                
        elif(1 == self.approximation):
            while(0 == areSameStrings(self.targetColumn, self.columnNames[self.index].split('.')[0])):
                self.index += 1

                if(self.index>=len(self.columnNames)):
                    raise StopIteration
                
        else:
            error.LOGGER.report("Invalid approximatioin flag.", error.LogType.ERROR)
            raise StopIteration

        self.index+=1

        return self.index-1
    
    @error.callStackRoutine
    def resetIndex(self):
        self.index = 0

    @error.callStackRoutine
    def setIndexTo(self, argIndex):
        self.index = argIndex

    @error.callStackRoutine
    def enableApprox(self):
        self.approximation = 1

    @error.callStackRoutine
    def disableApprox(self):
        self.approximation = 0
    
    @error.callStackRoutine
    def changeTargetColumnTo(self, argColumnToFind: str):
        self.targetColumn: str = argColumnToFind

    @error.callStackRoutine
    def changeTargetAndReset(self, argColumnToFind: str):
        self.changeTargetColumnTo(argColumnToFind)
        self.resetIndex()
        self.disableApprox()

    @error.callStackRoutine
    def findFirstIndex(self, argColumnToFind: str, argApprox: str):

        oldTarget = self.targetColumn
        oldIndex = self.index

        if('APPROX' == argApprox):
            self.enableApprox()
        elif('NOT_APPROX' == argApprox):
            self.disableApprox()
        else:
            error.LOGGER.report("Invalid approximation argument.", error.LogType.ERROR)
            return -1

        self.changeTargetAndReset(argColumnToFind)

        returnIndex =  self.__next__()

        self.disableApprox()
        self.setIndexTo(oldIndex)
        self.changeTargetColumnTo(oldTarget)

        return returnIndex

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

    if(int((hammingDistance/compareLength) <= hammingDistanceRatioCompareThreshold)):
        return 1

    return int((hammingDistance/compareLength) <= hammingDistanceRatioCompareThreshold)

@error.callStackRoutine
def ifStrMatchesAmong(argStr: str, argStrList: list) -> int:
    for string in argStrList:
        if(areSameStrings(argStr, string)):
            return 1
        
    else:
        return 0

@error.callStackRoutine
def isEmptyData(argData) -> int:

    if(None == argData):
        return 0
    elif(type(argData) == str):
        return int('.' == argData)
    else:
        return int(math.isnan(argData))
    
@error.callStackRoutine
def sortObjectBasedOn(argObjectList: list, argScoreList: list):

    if(len(argObjectList)!=len(argScoreList)):
        error.LOGGER.report("Length of two list should be equal.", error.LogType.ERROR)

    
    scoreListSorted = sorted(argScoreList, reverse = False)
    objectListSorted = []

    mapDict = {}

    for index in range(len(argScoreList)):
        if(argScoreList[index] in mapDict):
            mapDict[argScoreList[index]].append(argObjectList[index])
        else:
            mapDict[argScoreList[index]] = [argObjectList[index]]
    
    for score in scoreListSorted:
        objectListSorted.append(mapDict[score].pop())

    return objectListSorted

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




