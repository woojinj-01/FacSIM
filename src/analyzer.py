"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""
import os
import error
import cleaner
import institution
import util

"""
class Analyzer

Analyzing method added. 
This class encapsulates cleaners over different fields.


"""
class Analyzer:

    #instIdDict = {}
    #instDict = {}

    @error.callStackRoutine
    def __init__(self, argRankTypeList):
        
        self.__cleanerDict = {}
        self.instIdDict = {}
        self.instDict = {}

        self.__cleanedFlag = 0

        self.__rankTypeList = argRankTypeList

    @error.callStackRoutine
    def getRankTypeList(self):
        return self.__rankTypeList
    
    @error.callStackRoutine
    def __raiseCleanedFlag(self):
        self.__cleanedFlag = 1
    
    @error.callStackRoutine
    def __ifCleanedFlagNotRaised(self):
        return int(0 == self.__cleanedFlag)
        
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
    def getExistingInstitutionByName(self, argInstName) -> institution.Institution:

        for inst in util.getValuesListFromDict(self.instDict):

            if(util.areSameStrings(argInstName, inst.name)):
                return inst
            
        return None
    
    @error.callStackRoutine
    def getInstDict(self):
        return self.instDict
    
    @error.callStackRoutine
    def getInstMVRRankPhysics(self, argInst):
        return argInst.getRankAt('Physics', institution.RankType.SPRANK)
    
    @error.callStackRoutine
    def getInstMVRRankCS(self, argInst):
        return argInst.getRankAt('Computer Science', institution.RankType.SPRANK)
    
    
    @error.callStackRoutine
    def printAllInstitutions(self, **argOptions):

        optionList = ['based', 'field', 'granularity']

        optionDict = util.parseKwArgs(argOptions, optionList)

        instList = []

        for item in self.instDict.items():
            instList.append(item[1])
        
        if('auto' == optionDict['based']):
            itemDict = instList
        elif('mvr_rank' == optionDict['based']):
            if('physics' == optionDict['field']):
                itemDict = sorted(instList, key = self.getInstMVRRankPhysics)
            elif('computer science' == optionDict['field']):
                itemDict = sorted(instList, key = self.getInstMVRRankCS)
            elif('auto' == optionDict['field']):
                itemDict = sorted(instList, key = self.getInstMVRRankCS)
            else:
                error.LOGGER.report("Invalid field.", error.LogType.ERROR)
                return 0
            
        if('auto' == optionDict['granularity']):
            for item in itemDict:
                item.printInfo("all")
        elif('inst' == optionDict["granularity"]):
            for item in itemDict:
                item.printInfo("inst")
        elif('field' == optionDict["granularity"]):
            for item in itemDict:
                item.printInfo("field")
        elif('alumni' == optionDict["granularity"]):
            for item in itemDict:
                item.printInfo("all")
    
    @error.callStackRoutine
    def __queryCleanerDict(self, argField):

        return int(argField in self.__cleanerDict)

    @error.callStackRoutine
    def getCleanerFor(self, argField):

        if(util.isEmptyData(argField)):
            error.LOGGER.report("Attempt to generate Cleaner with empty value is suppressed",error.LogType.INFO)
            return None

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
        
        instIdDf = util.readFileFor(argFilePath, [util.FileExt.XLSX, util.FileExt.CSV])

        if(instIdDf.empty):
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

        targetDfDict = util.readFileFor(argFilePath, [util.FileExt.XLSX, util.FileExt.CSV])
        targetRow = None
        field = None

        for targetInst in util.getKeyListFromDict(targetDfDict):
            targetDf = targetDfDict[targetInst]
            rowIterator = util.rowIterator(targetDf.columns,'Degree')

            for numRow in range(len(targetDf.index)):

                targetRow = targetDf.iloc[numRow]
                field = util.gatherSimilarFields(targetRow[rowIterator.findFirstIndex('Department', 'APPROX')])

                cleaner = self.getCleanerFor(field)

                if(None != cleaner):
                    cleaner.cleanRow(targetRow, rowIterator)

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

        self.__raiseCleanedFlag()

        error.LOGGER.report("Data are Cleaned Now!", error.LogType.INFO)
    
    @error.callStackRoutine
    def exportVertexAndEdgeListFor(self, argField, argFileExtension: util.FileExt):

        if(self.__ifCleanedFlagNotRaised()):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)

        if(0 == self.__queryCleanerDict(argField)):
            error.LOGGER.report("Invalid Field Name", error.LogType.ERROR)
            return 0
        
        self.__cleanerDict[argField].exportVertexAndEdgeListAs(argFileExtension)

        return 1

    @error.callStackRoutine
    def exportVertexAndEdgeListForAll(self, argFileExtension: util.FileExt):

        if(self.__ifCleanedFlagNotRaised()):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return 0

        error.LOGGER.report(" ".join(["Exporting All Fields as", util.fileExtToStr(argFileExtension)]), error.LogType.INFO)

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):

            if(str == type(cleaner.field)):
                cleaner.exportVertexAndEdgeListAs(argFileExtension)

        error.LOGGER.report(" ".join(["Exported All Fields as", util.fileExtToStr(argFileExtension)]), error.LogType.INFO)
        return 1
    
    @error.callStackRoutine
    def calcGiniCoeffFor(self, argField):

        if(self.__ifCleanedFlagNotRaised()):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return 0

        if(0 == self.__queryCleanerDict(argField)):
            error.LOGGER.report("Invalid Field Name", error.LogType.ERROR)
            return 0

        return self.__cleanerDict[argField].calcGiniCoeff()
    
    @error.callStackRoutine
    def calcGiniCoeffForAll(self):

        if(self.__ifCleanedFlagNotRaised()):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return dict()
        
        error.LOGGER.report("Calculating Gini Coefficient for All Fields", error.LogType.INFO)

        giniCoeffDict = {}

        for field in util.getKeyListFromDict(self.__cleanerDict):
            giniCoeffDict[field] = self.calcGiniCoeffFor(field)

        error.LOGGER.report("Sucesssfully Calculated Gini Coefficient for All Fields!", error.LogType.INFO)
        return giniCoeffDict
    
    @error.callStackRoutine
    def calcRanksFor(self, argField):
        if(type(argField) != str):
            error.LOGGER.report("Field name should be a string", error.LogType.ERROR)
            return 0

        if(self.__ifCleanedFlagNotRaised()):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return 0
        
        if(0 == self.__queryCleanerDict(argField)):
            error.LOGGER.report("Invalid Field Name", error.LogType.ERROR)
            return 0
        
        error.LOGGER.report(' '.join(["Calculating Ranks for", argField]), error.LogType.INFO)

        return self.__cleanerDict[argField].calcRank()
    
    @error.callStackRoutine
    def calcRanksForAll(self):

        if(self.__ifCleanedFlagNotRaised()):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return 0
        
        error.LOGGER.report("Calculating MVR Ranks for All Fields", error.LogType.INFO)

        returnDict = {}

        for field in util.getKeyListFromDict(self.__cleanerDict):
            returnDict[field] = self.calcRanksFor(field)

        error.LOGGER.report("Sucesssfully Calculated MVR Ranks for All Fields!", error.LogType.INFO)

        return returnDict
    
    @error.callStackRoutine
    def calcMVRRankMoveFor(self, argField):

        if(type(argField) != str):
            error.LOGGER.report("Field name should be a string", error.LogType.ERROR)
            return 0

        if(self.__ifCleanedFlagNotRaised()):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return 0
        
        if(0 == self.__queryCleanerDict(argField)):
            error.LOGGER.report("Invalid Field Name", error.LogType.ERROR)
            return 0
        
        return self.getCleanerFor(argField).calcMVRMoveForAllAlumni()
    
    @error.callStackRoutine
    def calcMVRRankMoveForAll(self):

        if(self.__ifCleanedFlagNotRaised()):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return 0
        
        error.LOGGER.report("Calculating MVR Rank Movements for All Fields", error.LogType.INFO)

        returnValue = 1

        for field in util.getKeyListFromDict(self.__cleanerDict):
            returnValue = returnValue and self.calcMVRRankMoveFor(field)

        error.LOGGER.report("Sucesssfully Calculated MVR Rank Movements for All Fields!", error.LogType.INFO)

        return returnValue

    
    @error.callStackRoutine
    def calcAvgMVRMoveBasedOnGender(self, argGender: util.Gender):

        returnDict = {}

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            returnDict[cleaner.field] = cleaner.calcAvgMVRMoveBasedOnGender(argGender)

        return returnDict
    
    @error.callStackRoutine
    def calcAvgMVRMoveBasedOnGenderForField(self, argGender: util.Gender, argField: str):

        if(0 == self.__queryCleanerDict(argField)):
            error.LOGGER.report("Invalid Field.", error.LogType.ERROR)

        returnDict = {}

        returnDict[argField] = self.getCleanerFor(argField).calcAvgMVRMoveBasedOnGender(argGender)

        return returnDict

    @error.callStackRoutine
    def calcAvgMVRMoveForRange(self, argPercentLow: int, argPercentHigh: int):

        returnDict = {}

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            returnDict[cleaner.field] = cleaner.calcAvgMVRMoveForRange(argPercentLow, argPercentHigh)

        return returnDict

    @error.callStackRoutine
    def calcAvgMVRMoveForRangeForField(self, argPercentLow: int, argPercentHigh: int, argField: str):

        if(0 == self.__queryCleanerDict(argField)):
            error.LOGGER.report("Invalid Field.", error.LogType.ERROR)

        returnDict = {}

        returnDict[argField] = self.getCleanerFor(argField).calcAvgMVRMoveForRange(argPercentLow, argPercentHigh)

        return returnDict

    @error.callStackRoutine
    def plotRankMoveForGender(self, argGender: util.Gender, **argOptions):

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            cleaner.plotRankMoveForGender(argGender, **argOptions)

    @error.callStackRoutine
    def plotRankMoveCompareForGender(self, **argOptions):

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            cleaner.plotRankMoveCompareForGender(**argOptions)

    @error.callStackRoutine
    def plotRankMove(self, argLowerBound, argUpperBound):

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            cleaner.plotRankMove(argLowerBound, argUpperBound)

    @error.callStackRoutine
    def plotGenderRatio(self):

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            cleaner.plotGenderRatio()

    @error.callStackRoutine
    def plotNonKRFac(self):

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            cleaner.plotNonKRFac()

    @error.callStackRoutine
    def plotNonKRFacRatio(self):

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            cleaner.plotNonKRFacRatio()

if(__name__ == '__main__'):

    error.LOGGER.report("This Module is Not for Main Function", error.LogType.CRITICAL)
