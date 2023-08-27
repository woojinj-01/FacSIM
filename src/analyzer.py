"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""
import os
import error
import cleaner
import institution
import util
import career
import setting
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import waiter

class Analyzer:

    # ================================================================================== #
    # 0. Constructor
    # ================================================================================== #

    @error.callStackRoutine
    def __init__(self):
        
        self.__cleanerDict = {}
        # * Dictionary for managing Cleaners
        # Key: Field (str)
        # Value: Cleaner Object (cleaner.Cleaner)

        self.instIdDict = {}
        # * Dictionary for managing Institution IDs
        # Key: Institution Info (tuple)
        # Value: Institution ID (int)

        self.instDict = {}
        # * Dictionary for managing Institution Objects
        # Key: Institution ID (int)
        # Value: Institution Object (institution.Institution)

        self.flags = util.Flag(["Cleaned"])
        # * util.Flag Object to manage execution flow

        self.__nextPID = 1
        # * Integer that indicates next Personal ID

    # ================================================================================== #
    # 1. Methods for Management
    # ================================================================================== #

    # ====================================================== #
    # 1.a. Methods for Managing Cleaners
    # ====================================================== #

    @error.callStackRoutine
    def __queryCleanerDict(self, argField):
        # str -> int
        # Returns 1 if argField is among the keys of self.__cleanerDict, else 0

        return int(argField in self.__cleanerDict)

    @error.callStackRoutine
    def getCleanerFor(self, argField: str):
        # str -> None / cleaner.Cleaner
        # Returns cleaner.Cleaner Object if it is successful, else None

        if(util.isEmptyData(argField)):
            error.LOGGER.report("Attempt to generate Cleaner with empty value is suppressed", error.LogType.INFO)
            return None

        if(0 == self.__queryCleanerDict(argField)):
            self.__cleanerDict[argField] = cleaner.Cleaner(self, argField, self.isClosedSys())
            error.LOGGER.report("Got New Cleaner", error.LogType.INFO)
        
        return self.__cleanerDict[argField]

    # ====================================================== #
    # 1.b. Methods for Managing Institutions
    # ====================================================== #

    @error.callStackRoutine
    def __queryInstDictById(self, argInstId: int) -> int:
        # int -> int
        # Returns 1 if argInstId is among the keys of self.instDict, else 0 

        return int(argInstId in self.instDict)
    
    @error.callStackRoutine
    def getInstitution(self, argInstInfo: institution.InstInfo, argField: str) -> institution.Institution:
        # institution.InstInfo, str -> institution.Institution
        # Returns institution.Institution object based on argInstInfo and argField
        # ! Makes new institution.Institution object if it does not exist

        if(0 == self.__queryInstDictById(argInstInfo.instId)):
            self.instDict[argInstInfo.instId] = institution.Institution(argInstInfo)
            error.LOGGER.report("Got New Institution", error.LogType.INFO)

        self.instDict[argInstInfo.instId].getField(argField)
        
        return self.instDict[argInstInfo.instId]
    
    @error.callStackRoutine
    def getExistingInstitution(self, argInstId: int) -> institution.Institution:
        # int -> institution.Institution / None
        # Returns institution.Institution object based on argInstId
        # ! Does not make new institution.Institution object even if it does not exist

        if(0 == self.__queryInstDictById(argInstId)):
            error.LOGGER.report("Invalid Institution ID", error.LogType.WARNING)
            return None
        
        return self.instDict[argInstId]
    
    @error.callStackRoutine
    def getExistingInstitutionByName(self, argInstName: str) -> institution.Institution:
        # str -> institution.Institution / None
        # Returns institution.Institution object based on argInstName
        # ! Does not make new institution.Institution object even if it does not exist

        for inst in util.getValuesListFromDict(self.instDict):

            if(util.areSameStrings(argInstName, inst.name)):
                return inst
            
        return None
    
    @error.callStackRoutine
    def getInstDict(self):
        # -> dict
        # Returns self.instDict

        return self.instDict
    
    @error.callStackRoutine
    def getInstMVRRankPhysics(self, argInst):
        # institution.Institution -> int
        # Returns Rank at 'Physics' 

        return argInst.getRankAt('Physics', institution.RankType.SPRANK)
    
    @error.callStackRoutine
    def getInstMVRRankCS(self, argInst):
        # institution.Institution -> int
        # Returns Rank at 'Computer Science' 

        return argInst.getRankAt('Computer Science', institution.RankType.SPRANK)
    
    @error.callStackRoutine
    def printAllInstitutions(self, **argOptions):
        # dict -> 
        # Prints out various informations about current instutions

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

    # ====================================================== #
    # 1.c. Methods for Managing Institution IDs
    # ====================================================== #

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
    
    # ====================================================== #
    # 1.d. Methods for Managing Personal IDs
    # ====================================================== #

    @error.callStackRoutine
    def getNextPID(self):
        pID = self.__nextPID
        self.__nextPID += 1

        return pID
    
    # ====================================================== #
    # 1.e. Others
    # ====================================================== #
    
    @error.callStackRoutine
    def isClosedSys(self):
        return "CLOSED" == setting.PARAM["Basic"]["networkType"]
    
    # ================================================================================== #
    # 2. Methods for Analysis
    # ================================================================================== #

    # ====================================================== #
    # 2.a. Methods for Cleaning Data
    # ====================================================== #
    
    @error.callStackRoutine
    def __cleanDataForFile(self, argFilePath):

        targetDfDict = util.readFileFor(argFilePath, [util.FileExt.XLSX, util.FileExt.CSV])
        targetRow = None
        field = None

        print(f"Cleaning File: {argFilePath}")

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

            if('X_' == fileName[0:2]):  #masked file
                continue

            if('.xlsx' == os.path.splitext(fileName)[1]):
                self.__cleanDataForFile(os.path.join(targetDir, fileName))

        self.flags.raiseFlag("Cleaned")

        error.LOGGER.report("Data are Cleaned Now!", error.LogType.INFO)
    
    # ====================================================== #
    # 2.b. Methods for Setting Ranks
    # ====================================================== #

    @error.callStackRoutine
    def __setRanksFor(self, argField):
        if(type(argField) != str):
            error.LOGGER.report("Field name should be a string", error.LogType.ERROR)
            return 0

        if(not self.flags.ifRaised("Cleaned")):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return 0
        
        if(0 == self.__queryCleanerDict(argField)):
            error.LOGGER.report("Invalid Field Name", error.LogType.ERROR)
            return 0
        
        error.LOGGER.report(' '.join(["Calculating Ranks for", argField]), error.LogType.INFO)

        return self.__cleanerDict[argField].setRanks()
    
    @error.callStackRoutine
    def setRanks(self):

        if(not self.flags.ifRaised("Cleaned")):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return 0
        
        error.LOGGER.report("Calculating MVR Ranks for All Fields", error.LogType.INFO)

        returnDict = {}

        for field in util.getKeyListFromDict(self.__cleanerDict):
            returnDict[field] = self.__setRanksFor(field)

        error.LOGGER.report("Sucesssfully Calculated MVR Ranks for All Fields!", error.LogType.INFO)

        return returnDict
    
    # ====================================================== #
    # 2.c. Methods for Calculation of Metrics
    # ====================================================== #
    
    @error.callStackRoutine
    def calcAvgMVRMoveBasedOnGender(self, argGender: util.Gender, argDegTuple: tuple):

        returnDict = {}

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            returnDict[cleaner.field] = cleaner.calcAvgMVRMoveBasedOnGender(argGender, argDegTuple)

        return returnDict

    @error.callStackRoutine
    def calcAvgMVRMoveForRange(self, argRangeTuple: tuple, argDegTuple: tuple):

        returnDict = {}

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            returnDict[cleaner.field] = cleaner.calcAvgMVRMoveForRange(argRangeTuple, argDegTuple)

        return returnDict
    
    # ====================================================== #
    # 2.d. Methods for Plotting
    # ====================================================== #

    @error.callStackRoutine
    def plotRankMove(self, argRangeTuple, argDegTuple):

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            cleaner.plotRankMove(argRangeTuple, argDegTuple)

    @error.callStackRoutine
    def plotGenderRatio(self):

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            cleaner.plotGenderRatio()

    @error.callStackRoutine
    def plotNonKR(self, argDegTuple: tuple, argKROnly, argSizeOfCluster):

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):
            cleaner.plotNonKR(argDegTuple, argKROnly, argSizeOfCluster)

    @error.callStackRoutine
    def __plotLorentzCurveFor(self, argField, argDegTuple, argIntegrated):

        if(not self.flags.ifRaised("Cleaned")):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return 0

        if(0 == self.__queryCleanerDict(argField)):
            error.LOGGER.report("Invalid Field Name", error.LogType.ERROR)
            return 0

        return self.__cleanerDict[argField].plotLorentzCurve(argDegTuple, argIntegrated)
    
    @error.callStackRoutine
    def plotLorentzCurve(self, argDegTuple):

        if(not self.flags.ifRaised("Cleaned")):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return dict()
        
        error.LOGGER.report("Plotting Lorentz Curve for All Fields", error.LogType.INFO)

        giniCoeffDict = {}

        for field in util.getKeyListFromDict(self.__cleanerDict):
            giniCoeffDict[field] = self.__plotLorentzCurveFor(field, argDegTuple, 0)

        error.LOGGER.report("Sucesssfully Plotted for All Fields!", error.LogType.INFO)
        return giniCoeffDict

    @error.callStackRoutine
    def plotLorentzCurveIntegrated(self, argDegTuple):

        if(not self.flags.ifRaised("Cleaned")):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return dict()
        elif(not isinstance(argDegTuple, tuple)):
            error.LOGGER.report("Invalid argDegTuple.", error.LogType.ERROR)
            return 0
        
        error.LOGGER.report("Plotting Lorentz Curve for All Fields", error.LogType.INFO)

        srcDegType = argDegTuple[0]
        dstDegType = argDegTuple[1]

        giniCoeffDict = {}

        colorList = ['#4169E1', '#2E8B57', '#C71585']
        colorPointer = 0

        markerXCoList = [i for i in range(0, 101, 10)]

        font = {'family': 'Helvetica', 'size': 9}

        plt.rc('font', **font)
        plt.figure(figsize=(7,5), dpi=200)

        titleStr = f"Lorentz Curve on {srcDegType.toStr('label')}_{dstDegType.toStr('label')} Production (Integrated)"
        ylabelStr = f"Cumulative Ratio Over Total {srcDegType.toStr('label')}_{dstDegType.toStr('label')} Production (Unit: Percentage)"

        plt.title(titleStr)
        plt.xlabel("Cumulative Ratio Over Total Number of Institutions (Unit: Percentage)")
        plt.ylabel(ylabelStr)

        plt.xlim(np.float32(0), np.float32(100))
        plt.ylim(np.float32(0), np.float32(100))

        for field in util.getKeyListFromDict(self.__cleanerDict):
            (giniCoeffDict[field], xCoList, yCoList, baseList) = self.__plotLorentzCurveFor(field, argDegTuple, 1)

            plt.plot(xCoList, yCoList, color = colorList[colorPointer], linewidth = 1.5, label = field)
            plt.scatter(markerXCoList, [util.sampleLinePlot(xCoList, yCoList, index) for index in markerXCoList], \
                        c = colorList[colorPointer], s = 20, clip_on = False, alpha = 1)

            plt.plot(baseList, baseList, color = 'black', linewidth = 1)

            colorPointer += 1

        plt.legend()
        
        figPath = waiter.WAITER.getPlotPath(f"{srcDegType.toStr('label')}_{dstDegType.toStr('label')} Production", "LorentzCurve", "Integrated")
        #figPath = util.getPlotPath(f"{srcDegType.toStr('label')}_{dstDegType.toStr('label')} Production", "LorentzCurve", "Integrated")
        plt.savefig(figPath)
        plt.clf()

        error.LOGGER.report("Sucesssfully Plotted for All Fields!", error.LogType.INFO)
        return giniCoeffDict
    

    # ====================================================== #
    # 2.e. Methods for Extracting Network Data
    # ====================================================== #

    @error.callStackRoutine
    def exportVertexAndEdgeList(self, argFileExtension: util.FileExt):

        if(not self.flags.ifRaised("Cleaned")):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return 0

        error.LOGGER.report(" ".join(["Exporting All Fields as", util.fileExtToStr(argFileExtension)]), error.LogType.INFO)

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):

            if(str == type(cleaner.field)):
                cleaner.exportVertexAndEdgeList(argFileExtension)

        error.LOGGER.report(" ".join(["Exported All Fields as", util.fileExtToStr(argFileExtension)]), error.LogType.INFO)
        return 1

    @error.callStackRoutine
    def exportRankList(self, argTargetDeg, argFileExtension: util.FileExt):

        if(not self.flags.ifRaised("Cleaned")):
            error.LOGGER.report("Attempt denied. Data are not cleaned yet.", error.LogType.ERROR)
            return 0

        error.LOGGER.report(" ".join(["Exporting RankList for All Fields as", util.fileExtToStr(argFileExtension)]), error.LogType.INFO)

        for cleaner in util.getValuesListFromDict(self.__cleanerDict):

            if(str == type(cleaner.field)):
                cleaner.exportRankList(argTargetDeg, argFileExtension)

        error.LOGGER.report(" ".join(["Exported RankList for All Fields as", util.fileExtToStr(argFileExtension)]), error.LogType.INFO)
        return 1
    

if(__name__ == '__main__'):

    error.LOGGER.report("This Module is Not for Main Function", error.LogType.CRITICAL)
