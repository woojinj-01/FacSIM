"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""

import pandas as pd
import datetime as dt
import os
import error
import cleaner
import institution
import util
import logging

"""
class Analyzer

Analyzing method added. 
This class encapsulates cleaners over different fields.


"""
class Analyzer:

    #instIdDict = {}
    #instDict = {}

    @error.callStackRoutine
    def __init__(self):

        self.__cleanerDict = {}
        self.instIdDict = {}
        self.instDict = {}
        
    @error.callStackRoutine
    def __queryInstDictById(self, argInstId):

        return int(argInstId in self.instDict)
    
    @error.callStackRoutine
    def getInstitution(self, argInstInfo: institution.InstInfo, argField: str) -> institution.Institution:

        if(0 == self.__queryInstDictById(argInstInfo.instId)):
            self.instDict[argInstInfo.instId] = institution.Institution(argInstInfo)
            error.LOGGER.report("Got New Institution", error.LogType.INFO)

        self.instDict[argInstInfo.instId].getField(argField)
        
        return self.instDict[argInstInfo.instId]
    
    @error.callStackRoutine
    def getExistingInstitution(self, argInstId) -> institution.Institution:

        if(0 == self.__queryInstDictById(argInstId)):
            error.LOGGER.report("Invalid Institution ID", error.LogType.WARNING)
            return None
        
        return self.instDict[argInstId]
    
    @error.callStackRoutine
    def getInstDict(self):
        return self.instDict
    
    @error.callStackRoutine
    def printAllInstitutions(self):

        for item in sorted(self.instDict.items()):
            print(type(item[1]))
            item[1].printInfo()
    
    @error.callStackRoutine
    def __queryCleanerDict(self, argField):

        return int(argField in self.__cleanerDict)

    @error.callStackRoutine
    def getCleanerFor(self, argField):

        if(0 == self.__queryCleanerDict(argField)):
            self.__cleanerDict[argField] = cleaner.Cleaner(self, argField)
            error.LOGGER.report("Got New Cleaner", error.LogType.INFO)
        
        return self.__cleanerDict[argField]

    @error.callStackRoutine
    def getInstIdFor(self, argInstInfo: institution.InstInfo):

        keyTuple = argInstInfo.returnKeyTuple()

        if(keyTuple in self.instIdDict):
            return self.instIdDict[keyTuple]
        else:
            newId = len(self.instIdDict)+1
            self.instIdDict[keyTuple] = newId
            error.LOGGER.report("New Inst ID Allocated", error.LogType.INFO)

            return newId
    
    @error.callStackRoutine
    def getInstIdForInit(self, argKeyTuple: tuple):

        if(argKeyTuple in self.instIdDict):
            return 0
        else:
            newId = len(self.instIdDict)+1
            self.instIdDict[argKeyTuple] = newId

            return 1

    @error.callStackRoutine
    def printInstIdDict(self):

        print(self.instIdDict)

    @error.callStackRoutine
    def loadInstIdDictFrom(self, argFilePath):
        
        instIdDf = util.readFileFor(argFilePath, ['.xlsx', '.csv'])

        if(0 == instIdDf):
            error.LOGGER.report("Failed to Initialize InstID Dictionary", error.LogType.WARNING)
            return 0
        
        returnValue = 1

        for numRow in range(len(instIdDf.index)):
            returnValue &= self.getInstIdForInit((instIdDf.iloc[numRow][0], instIdDf.iloc[numRow][1], instIdDf.iloc[numRow][2]))

        if(0 == returnValue):
            error.LOGGER.report("Failed to Initialize InstID Dictionary", error.LogType.WARNING)
        return returnValue
    
    @error.callStackRoutine
    def __cleanDataForFile(self, argFilePath):

        targetDf = pd.read_excel(argFilePath, engine = "openpyxl")
        rowIterator = util.rowIterator(targetDf.columns,'Degree')

        targetRow = None
        field = None

        for numRow in range(len(targetDf.index)):

            targetRow = targetDf.iloc[numRow]
            field = targetRow[7]

            self.getCleanerFor(field).cleanRow(targetRow, rowIterator)

        error.LOGGER.report(" ".join([argFilePath, "is Cleaned"]), error.LogType.INFO)

    @error.callStackRoutine
    def cleanData(self):
        error.LOGGER.report("Start Cleaning Data", error.LogType.INFO)

        targetDir = '../dataset/dirty'

        if(False == os.path.isdir(targetDir)):
            error.LOGGER.report(" ".join(["No Directory Named", targetDir]), error.LogType.CRITICAL)

        for fileName in os.listdir(targetDir):
            if('~$' == fileName[0:2]):  #getting rid of cache file
                continue

            if('.xlsx' == os.path.splitext(fileName)[1]):
                self.__cleanDataForFile(os.path.join(targetDir, fileName))


        error.LOGGER.report("Data are Cleaned Now!", error.LogType.INFO)
    
    @error.callStackRoutine
    def exportVertexAndEdgeListFor(self, argField, argFileExtension):
        if(0 == self.__queryCleanerDict(argField)):
            error.LOGGER.report("Invalid Field Name", error.LogType.WARNING)
            return 0
        
        self.__cleanerDict[argField].exportVertexAndEdgeListAs(argFileExtension)

        return 1

    @error.callStackRoutine
    def exportVertexAndEdgeListForAll(self, argFileExtension):
        error.LOGGER.report(" ".join(["Exporting All Fields as", argFileExtension]), error.LogType.INFO)

        for cleaner in list(self.__cleanerDict.values()):

            if(str == type(cleaner.field)):
                cleaner.exportVertexAndEdgeListAs(argFileExtension)

        error.LOGGER.report(" ".join(["Exported All Fields as", argFileExtension]), error.LogType.INFO)
        return 1
    
    @error.callStackRoutine
    def calcGiniCoeffFor(self, argField):
        if(0 == self.__queryCleanerDict(argField)):
            error.LOGGER.report("Invalid Field Name", error.LogType.WARNING)
            return 0

        return self.__cleanerDict[argField].calcGiniCoeff()
    
    @error.callStackRoutine
    def calcGiniCoeffForAll(self):
        error.LOGGER.report("Calculating Gini Coefficient for All Fields", error.LogType.INFO)

        giniCoeffDict = {}

        for field in list(self.__cleanerDict.keys()):
            giniCoeffDict[field] = self.calcGiniCoeffFor(field)

        error.LOGGER.report("Sucesssfully Calculated Gini Coefficient for All Fields!", error.LogType.INFO)
        return giniCoeffDict
    
    @error.callStackRoutine
    def calcMVRRAnkFor(self, argField):
        if(0 == self.__queryCleanerDict(argField)):
            error.LOGGER.report("Invalid Field Name", error.LogType.WARNING)
            return 0

        return self.__cleanerDict[argField].calcMVRRank()
    
    @error.callStackRoutine
    def calcMVRRAnkForAll(self):
        error.LOGGER.report("Calculating MVR Ranks for All Fields", error.LogType.INFO)

        returnValue = 1

        for field in list(self.__cleanerDict.keys()):
            returnValue = returnValue and self.calcMVRRAnkFor(field)

        error.LOGGER.report("Sucesssfully Calculated MVR Ranks for All Fields!", error.LogType.INFO)

        return returnValue

        

if(__name__ == '__main__'):

    raise Exception("This Module is Not for Main Function")

    
        

        


        

    
