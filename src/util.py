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
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import sys
import configparser

global TYPOHISTORY

lengthCompareThreshold = 3
hammingDistanceRatioCompareThreshold = 0.25

goldDateStr = None
goldTimeStr = None

fieldClassificationDict = {'Physics': ['Physics', 'Applied Physics', 'Electrical and Biological Physics', \
                                       'Physics and Astronomy', 'Physics and Engineering'], \
                            'Computer Science': ['Artificial Intelligence', 'Autonomous Things Intelligence', \
                                        'College of Computing', 'Computer and Information Engineering', \
                                        'Computer and Software Engineering', 'Computer and Telecommunications Engineering', \
                                        'Computer Convergence Software', 'Computer Engineering', 'Computer Engineerinng', \
                                        'Computer Information Communication', 'Computer Informatoin Engineering', \
                                        'Computer Science and Engineering', 'Computer Science and Information Engineering', \
                                        'Computer Science and Software', 'Computer Science', 'Computer Software', \
                                        'Computer', 'ComputerScience', 'Computing', 'Department of Computer Science and Engineering', \
                                        'Electrical Engineering and Computer Science', 'Information and Communication Engineering', \
                                        'Information Convergence', 'Mobile Systems Engineering', 'Multimedia Engineering', \
                                        'Software and Computer Engineering', 'Software Science', 'Software'],
                            'Biology' : [ 'Microbiology', 'Bioscience', 'biosystemscience', 'Biology and Chemistry', 'life sciences', \
                                        'System Biotechnology', 'Life science', 'Biological Sciences', 'Life Sciences', \
                                        'Biology', 'marine lifesciences', 'Department of Biomedical Science', 'APPLIED BIOLOGY.', \
                                        'Agricultural Biology', 'Bio Convergence Science', 'Molecular Biology', 'biological sciences', \
                                        'Chemistry & Life Science', 'Biotechnology', 'Systems Biology', 'Biological Science and Technology', \
                                        'Biology', 'life science', 'microbiology', 'Department of Life Science', 'Department of Biological Sciences', \
                                        'Genetic Biotechnology', 'Marine Molecular Bioscience', 'biotechnology',]}

class RankType(Enum):

    SPRANK = auto()
    JRANK = auto()

    def toStr(self, argRepType):
        if("camelcase" == argRepType):
            match self:
                case RankType.SPRANK:
                    return "SpringRank"
                case RankType.JRANK:
                    return "Joongang-ilbo Rank"
                case _:
                    pass

        elif("abbrev" == argRepType):
            return self.name.lower()
        
        else:
            error.LOGGER.report("Wrong argRepType.", error.LogType.ERROR)
            
class HistType(Enum):
    LOCAL = auto()
    GLOBAL = auto()

class Flag:
    def __init__(self, argLabelList) -> None:

        if(not isinstance(argLabelList, list)):
            return None
        elif(any(not isinstance(label, str) for label in argLabelList)):
             return None
            
        self.numLabel = len(argLabelList)
        self.labelList = argLabelList

        self.flagList = [False] * self.numLabel

    def __query(self, argLabel):
        if(argLabel in self.labelList):
            return self.labelList.index(argLabel)
        else:
            return None
        
    def raiseFlag(self, argLabel):
        
        index = self.__query(argLabel)

        if(None == index):
            return None
        
        self.flagList[index] = True

        return self.flagList[index]
    
    def lowerFlag(self, argLabel):
        
        index = self.__query(argLabel)

        if(None == index):
            return None
        
        self.flagList[index] = False

        return self.flagList[index]

    def ifRaised(self, argLabel):
        
        index = self.__query(argLabel)

        if(None == index):
            return None
        
        return self.flagList[index]

class TypoHistory:
    def __init__(self, argAbled):

        globalHistPath = "../ini/globalTypoHistory.ini"

        globalConfig = None

        if(not os.path.exists(globalHistPath)):
            (os.open(globalHistPath, 'w')).close()

        globalConfig = configparser.ConfigParser(allow_no_value = True)
        globalConfig.read(globalHistPath, encoding='utf-8')

        self.globalConfig = globalConfig
        self.localConfig = configparser.ConfigParser(allow_no_value = True)

        self.localHistoryDict = {}

        match argAbled:
            case "ENABLE":
                self.abled = 1
            case "DISABLE":
                self.abled = 0
            case _:
                self.abled = 1
    
    @error.callStackRoutine
    def __addTripletToConfig(self, argConfigType, argSrcStr1, argSrcStr2, argDstStr):

        if(HistType.LOCAL == argConfigType):
            config = self.localConfig
        elif(HistType.GLOBAL == argConfigType):
            config = self.globalConfig
        else:
            error.LOGGER.report("argConfigType: Invalid config type.", error.LogType.ERROR)
            return None
            
        if(not config.has_section(argSrcStr1)):
            config.add_section(argSrcStr1)

        config.set(argSrcStr1, argSrcStr2, argDstStr)

    @error.callStackRoutine
    def __readTripletFromConfig(self, argConfigType, argSrcStr1, argSrcStr2):
        if(HistType.LOCAL == argConfigType):
            config = self.localConfig
        elif(HistType.GLOBAL == argConfigType):
            config = self.globalConfig
        else:
            error.LOGGER.report("argConfigType: Invalid config type.", error.LogType.ERROR)
            return None

        if(config.has_section(argSrcStr1) and config.has_option(argSrcStr1, argSrcStr2)):
            return config.get(argSrcStr1, argSrcStr2)
        else:
            return None
        
    @error.callStackRoutine
    def flush(self):

        globalHistPath = "../ini/globalTypoHistory.ini"

        self.globalConfig.update(self.localConfig)

        with open(globalHistPath, 'w') as globalTypoHistroy:
            self.globalConfig.write(globalTypoHistroy)

    @error.callStackRoutine
    def writeHistory(self, argSrcStr1: str, argSrcStr2: str, argDstStr: str):
        if(not self.abled):
            error.LOGGER.report("User-interactive typo correction is disabled", error.LogType.DEBUG)
            return 0
        
        self.__addTripletToConfig(HistType.LOCAL, argSrcStr1, argSrcStr2, argDstStr)

        return 1
    
    @error.callStackRoutine
    def readHistory(self, argSrcStr1: str, argSrcStr2: str):
        if(not self.abled):
            error.LOGGER.report("User-interactive typo correction is disabled", error.LogType.DEBUG)
            return 0

        globalHistRead = self.__readTripletFromConfig(HistType.GLOBAL, argSrcStr1, argSrcStr2)

        if(None != globalHistRead):
            return globalHistRead
        
        localHistRead = self.__readTripletFromConfig(HistType.LOCAL, argSrcStr1, argSrcStr2)

        if(None != localHistRead):
            return localHistRead
        
        return None
        
#class FileExt
#every file extension should be represented using this Enum class
class FileExt(Enum):
    CSV = auto()
    XLSX = auto()
    PNG = auto()
    JPG = auto()
    TXT = auto()

#converts a string to a FileExt instance 
# ex) argument: '.csv', return value: FileExt.CSV
@error.callStackRoutine
def strToFileExt(argFileExtStr: str) -> FileExt:
    match argFileExtStr:
        case '.csv':
            return FileExt.CSV
        case '.xlsx':
            return FileExt.XLSX
        case '.png':
            return FileExt.PNG
        case '.jpg':
            return FileExt.JPG
        case '.txt':
            return FileExt.TXT
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
        case FileExt.PNG:
            return '.png'
        case FileExt.JPG:
            return '.jpg'
        case FileExt.TXT:
            return '.txt'
        case _:
            error.LOGGER.report("Invalid Argument", error.LogType.WARNING)
            return None

@error.callStackRoutine
def getPlotPath(argSubject, argPlotType, argField, *argOthers) -> str:

    if(None != goldDateStr):
        currentDateStr = goldDateStr
    else:
        currentDateStr = dt.datetime.now().strftime("%Y_%m_%d")
    
    if(None != goldTimeStr):
        currentTimeStr = goldTimeStr
    else:
        currentTimeStr = dt.datetime.now().strftime("%H:%M:%S")

    targetDir = "../plot/" + currentDateStr + "/" + currentTimeStr

    if(not os.path.exists(targetDir)):
        os.makedirs(targetDir)

    plotNameList = [currentTimeStr, argSubject, argPlotType, argField]

    for word in argOthers:
        plotNameList.append(str(word))

    return targetDir + "/" + '_'.join(plotNameList) + ".png"

@error.callStackRoutine
def getCleanedFilePath(argSubject, argField, argFileExt: FileExt, *argOthers) -> str:

    if(None != goldDateStr):
        currentDateStr = goldDateStr
    else:
        currentDateStr = dt.datetime.now().strftime("%Y_%m_%d")
    
    if(None != goldTimeStr):
        currentTimeStr = goldTimeStr
    else:
        currentTimeStr = dt.datetime.now().strftime("%H_%M_%S")

    targetDir = "../dataset/cleaned/" + argField + "/" + currentDateStr + "/" + currentTimeStr

    if(not os.path.exists(targetDir)):
        os.makedirs(targetDir)

    cleanedFileNameList = [argSubject, argField, currentDateStr, currentTimeStr]

    for word in argOthers:
        cleanedFileNameList.append(str(word))

    return targetDir + "/" + '_'.join(cleanedFileNameList) + fileExtToStr(argFileExt)

@error.callStackRoutine
def getResultFilePath(argFileExt: FileExt) -> str:
    currentDateStr = dt.datetime.now().strftime("%Y_%m_%d")
    currentTimeStr = dt.datetime.now().strftime("%H_%M_%S")

    targetDir = "../result/" + currentDateStr

    if(not os.path.exists(targetDir)):
        os.makedirs(targetDir)

    resultFileNameList = ["Result", currentDateStr, currentTimeStr]

    return targetDir + "/" + '_'.join(resultFileNameList) + fileExtToStr(argFileExt)

@error.callStackRoutine
def getRidOfTie(argRankList: list) -> list:

    tieRankCountDict = {}
    newRankDict = {}

    returnRankList = []
    
    for rank in argRankList:
        if(rank in tieRankCountDict):
            tieRankCountDict[rank] +=1
        else:
            tieRankCountDict[rank] = 1

    for rankI in argRankList:
        if(rankI in newRankDict):
            newRankDict[rankI].append(newRankDict[rankI][-1] + 1)
            continue
        
        newRank = 1
        
        for rankJ in range(1, rankI):

            if(rankJ in tieRankCountDict):
                newRank += tieRankCountDict[rankJ]

        newRankDict[rankI] = [newRank]
    
    
    for rank in argRankList:
        returnRankList.append(newRankDict[rank].pop(0))

    return returnRankList

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
        
@error.callStackRoutine
def gatherSimilarFields(argField: str) -> str:

    for parentField in getKeyListFromDict(fieldClassificationDict):
        for childField in fieldClassificationDict[parentField]:
            if(areSameStrings(argField, childField)):
                return parentField
            
    return argField

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
    
@error.callStackRoutine
def userSelectsString(argStr1: str, argStr2: str) -> str:

    if(argStr1 == argStr2):
        return argStr1
    elif(not TYPOHISTORY.abled):
        return argStr1
    elif(isEmptyData(argStr1)):
        return argStr1
    
    history = TYPOHISTORY.readHistory(argStr1, argStr2)

    if(None != history):
        return history
    
    originalStream = sys.stdout
    sys.stdout = sys.__stdout__

    print("========== Typo Correction ==========")
        
    while(1):

        print("Press 1 to Use", argStr1)
        print("Press 2 to Merge", argStr1, "To", argStr2)

        select = int(input())

        dstStr = argStr1 if 1 == select else argStr2 if 2 == select else None

        if(None == dstStr):
            print("Wrong Key. Press 1 or 2")
        else:
            TYPOHISTORY.writeHistory(argStr1, argStr2, dstStr)
            sys.stdout = originalStream
            return dstStr


#better way for string compare (strongly recommended)
@error.callStackRoutine
def areSameStrings(argStr1: str, argStr2: str) -> int:

    if(type(argStr1) != str):
        return 0
    
    if(type(argStr2) != str):
        return 0
    
    compactStr1 = argStr1.replace(" ", "")
    compactStr2 = argStr2.replace(" ", "")

    if(compactStr1.lower() == compactStr2.lower()):
        return 1
    elif(compactStr1.lower() in compactStr2.lower()):
        return 1
    elif(compactStr1.lower() in compactStr2.lower()):
        return 1
    
    hammingStr1 = compactStr1.replace("university", "")
    hammingStr2 = compactStr2.replace("university", "")

    if(lengthCompareThreshold <= abs(len(hammingStr1) - len(hammingStr2))):
        return 0

    compareLength = min([len(hammingStr1), len(hammingStr2)])
    hammingDistance = 0 

    for index in range(compareLength):
        if(hammingStr1[index] != hammingStr2[index]):
            hammingDistance += 1

    if(int((hammingDistance/compareLength) <= hammingDistanceRatioCompareThreshold)):
        return 1
    
    return 0

@error.callStackRoutine
def areSameStringsInteractive(argStr1: str, argStr2: str) -> int:

    if(type(argStr1) != str):
        return 0
    
    if(type(argStr2) != str):
        return 0
    
    compactStr1 = argStr1.replace(" ", "")
    compactStr2 = argStr2.replace(" ", "")
    
    if(compactStr1.lower() == compactStr2.lower()):
        return userSelectsString(argStr1, argStr2)
    elif(compactStr1.lower() in compactStr2.lower()):
        return userSelectsString(argStr1, argStr2)
    elif(compactStr1.lower() in compactStr2.lower()):
        return userSelectsString(argStr1, argStr2)
    elif(compactStr1 in compactStr2):
        return userSelectsString(argStr1, argStr2)
    elif(compactStr2 in compactStr1):
        return userSelectsString(argStr1, argStr2)
    
    hammingStr1 = compactStr1.replace("university", "")
    hammingStr2 = compactStr2.replace("university", "")

    if(lengthCompareThreshold <= abs(len(hammingStr1) - len(hammingStr2))):
        return 0

    compareLength = min([len(hammingStr1), len(hammingStr2)])
    hammingDistance = 0

    for index in range(compareLength):
        if(hammingStr1[index] != hammingStr2[index]):
            hammingDistance += 1

    if(int((hammingDistance/compareLength) <= hammingDistanceRatioCompareThreshold)):
        return userSelectsString(argStr1, argStr2)
    
    return 0

@error.callStackRoutine
def ifStrMatchesAmong(argStr: str, argStrList: list) -> int:
    for string in argStrList:
        if(areSameStrings(argStr, string)):
            return 1
        
    return 0

@error.callStackRoutine
def appendIfNotIn(argElem, argList):

    value = ifStrMatchesAmongInteractive(argElem, argList)

    if(0 == value):
        argList.append(argElem)
        return argElem
    else:
        return value
    
@error.callStackRoutine
def ifStrMatchesAmongInteractive(argStr: str, argStrList: list) -> int:

    for string in argStrList:
        if(areSameStrings(argStr, string)):
            return userSelectsString(argStr, string)
        
    return 0

@error.callStackRoutine
def isEmptyData(argData) -> int:

    if(None == argData):
        return 1
    elif(type(argData) == str):
        return int('.' == argData) or '' == argData
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
    rank = len(argScoreList)

    for index in range(len(argScoreListSorted)):

        score = argScoreListSorted[index]

        scoreToRankMap[score] = rank

        if(prevScore <= score):
            rank -= 1
        
        prevScore = score


    for score in argScoreList:
        rankList.append(scoreToRankMap[score])

    return rankList 

@error.callStackRoutine
def calGiniCoeff(argList):

    if(0 == len(argList)):
        return None
    
    argList.sort()

    totalNum = len(argList)
    totalSum = sum(argList)
    percentage_delta = np.float32((1 / (totalNum-1) * 100))
    height_1 = np.float32(argList[0]/totalSum*100)
    height_2 = np.float32(argList[0]+argList[1]/totalSum*100)

    area_AnB = (100 * 100)/2
    area_B = 0

    #for plotting
    xCoList = []
    yCoList = []
    baseList = []

    for i in range(totalNum-1):
        area_B += np.float32(percentage_delta * (height_1 + height_2)/2)

        xCoList.append(percentage_delta * i)
        yCoList.append(height_1)
        baseList.append(percentage_delta * i)

        if(totalNum-2 != i):
            height_1 = height_2
            height_2 += np.float32(argList[i+2]/totalSum*100)

    giniCoeff = np.float32((area_AnB - area_B)/area_AnB)
    
    xCoList.append(np.float32(100))
    yCoList.append(np.float32(100))
    baseList.append(np.float32(100))

    return (giniCoeff, xCoList, yCoList, baseList)
        
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
            return pd.read_csv(argFilePath, sep ='\t')
        case FileExt.XLSX:
            dfDict = pd.read_excel(argFilePath, sheet_name=None)

            if(1 == len(dfDict)):   #if the file contains a single sheet
                return list(dfDict.values())[0]   #returns a dataframe
            else:                   #else
                return pd.read_excel(argFilePath, sheet_name=None)  #returns a dictionary of dataframes
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

@error.callStackRoutine
def parseKwArgs(argKwArgs: dict, argKeyWordList: list) -> dict:

    returnDict = {}

    for keyword in getKeyListFromDict(argKwArgs):
        if(keyword not in argKeyWordList):
            error.LOGGER.report("Invalid arguments will be ignored", error.LogType.WARNING)

    for keyword in argKeyWordList:
        if(keyword in argKwArgs):
            returnDict[keyword] = argKwArgs[keyword]
        else:
            returnDict[keyword] = 'auto'

    return returnDict

@error.callStackRoutine
def isLabelOfEnum(argEnumClass, argLabel):
    return any(argLabel == member.name for member in argEnumClass)

@error.callStackRoutine
def getStrLineEq(argFirstPointTuple, argSecondPointTuple):
    def wrapper(argXCo):

        lowXCo = argFirstPointTuple[0]
        lowYCo = argFirstPointTuple[1]

        highXCo = argSecondPointTuple[0]
        highYCo = argSecondPointTuple[1]

        gradient = (highYCo - lowYCo) / (highXCo - lowXCo)

        return gradient * (argXCo - lowXCo) + lowYCo
    return wrapper

@error.callStackRoutine
def sampleLinePlot(argXCoList, argYCoList, argXCoTosample):

    if(any(not isinstance(targetList, list) for targetList in [argXCoList, argYCoList])):
        error.LOGGER.report("argXCoList and argYCoList should be list type", error.LogType.ERROR)
        return None
    elif(sorted(argXCoList) != argXCoList):
        error.LOGGER.report("argXCoList must be sorted", error.LogType.ERROR)
        return None
    elif(len(argXCoList) != len(argYCoList)):
        error.LOGGER.report("argXCoList and argYCoList should be same length.", error.LogType.ERROR)
        return None
    elif(len(argXCoList) < 2):
        error.LOGGER.report("There should be at least two data points.", error.LogType.ERROR)
        return None
    
    if(argXCoTosample in argXCoList):
        return argYCoList[argXCoList.index(argXCoTosample)]
    
    elif(min(argXCoList) > argXCoTosample):
        lowXIndex = 0
        highXIndex = lowXIndex + 1

    elif(max(argXCoList) < argXCoTosample):
        lowXIndex = len(argXCoList) - 2
        highXIndex = lowXIndex + 1
    else:
        for lowXIndex in range(len(argXCoList)-1):

            highXIndex = lowXIndex + 1

            if(argXCoList[lowXIndex] < argXCoTosample < argXCoList[highXIndex]):
                break

    lowXCo = argXCoList[lowXIndex]
    lowYCo = argYCoList[lowXIndex]

    highXCo = argXCoList[highXIndex]
    highYCo = argYCoList[highXIndex]

    lineEq = getStrLineEq((lowXCo, lowYCo), (highXCo, highYCo))

    return lineEq(argXCoTosample)

@error.callStackRoutine
def listsToDataFrame(argColumnList, *argLists):

    if(1 != len(argLists) or not isinstance(argLists[0], list) or 0 == len(argLists[0])):
        return None
    else:
        lists = argLists[0]

    if(not isinstance(argColumnList, list)):
        return None
    elif(any(not isinstance(targetList, list) for targetList in lists)):
        return None
    elif(len(argColumnList) != len(lists)):
        return None
    elif(any(len(targetList) != len(lists[0]) for targetList in lists)):
        return None

    
    dataDict = {}

    for index in range(len(argColumnList)):
        dataDict[argColumnList[index]] = lists[index]

    return pd.DataFrame(dataDict)

@error.callStackRoutine
def matrixToDataFrame(argMatrix, argRowLabelList, argColLabelList):
    return pd.DataFrame(data = argMatrix, columns= argColLabelList, index=argRowLabelList)

@error.callStackRoutine
def calcSparsity(argMatrix):
    return 1.0 - (np.count_nonzero(argMatrix) / float(argMatrix.size))

if(__name__ == '__main__'):

    error.LOGGER.report("This Module is Not for Main Function", error.LogType.CRITICAL)




