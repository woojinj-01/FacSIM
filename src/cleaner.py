"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""

import pandas as pd
import error
import util
import institution
import numpy as np
import SpringRank as sp
import math
import status
import os

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad

class CleanerFlag():

    @error.callStackRoutine
    def __init__(self):
        self.MVRRankCalculated = False
        self.MVRRankMoveCalculated = False
        self.giniCoeffCalculated = False

    @error.callStackRoutine
    def raiseMVRRankCalculated(self):
        self.MVRRankCalculated = True

    @error.callStackRoutine
    def ifMVRRankCalculated(self):
        return self.MVRRankCalculated
    
    @error.callStackRoutine
    def raiseMVRRankMoveCalculated(self):
        self.MVRRankMoveCalculated = True

    @error.callStackRoutine
    def ifMVRRankMoveCalculated(self):
        return self.MVRRankMoveCalculated
    
    @error.callStackRoutine
    def raiseGiniCoeffCalculated(self):
        self.giniCoeffCalculated = True

    @error.callStackRoutine
    def ifGiniCoeffCalculated(self):
        return self.giniCoeffCalculated


"""
class Cleaner

For every single .xlsx file in dataset_dirty directory, 
clean them and make two output .csv files (vertex list and edge list) in dataset_cleaned directory
Cleaner is allocated 'per field', by class analyzer.Analyzer. 
Cleaner is also responsible for any calculation such as Gini Coefficient or MVR Ranking, etc.

"""
class Cleaner():

    """
    __init__(self)

    Constructor of class Cleaner. You can set columns of output files by modifying the constructor.
    """
    @error.callStackRoutine
    def __init__(self, argAnalyzer, argField):

        self.analyzer= argAnalyzer

        # Field Name
        self.field = argField

        # Dataframe of output vertex list
        self.__vertexList = pd.DataFrame(columns = ['# u', 'Institution', 'Region', 'Country'])

        # Dataframe of output edge list
        self.__edgeList = pd.DataFrame(columns = ['# u', '# v', 'gender'])

        # Dictonary holding institution ids, which are matched to institutions one by one.
        self.__localInstIdList = []

        # Flags.
        self.flags = CleanerFlag()

    @error.callStackRoutine
    def getTotalNumInstInField(self):
        return len(self.__localInstIdList)
    
    @error.callStackRoutine
    def __addToLocalInstIdList(self, argInstId):

        if(argInstId in self.__localInstIdList):
            return 0
        else:
            self.__localInstIdList.append(argInstId)
            return 1

    @error.callStackRoutine
    def cleanRow(self, argTargetRow, argRowIterator):
        error.LOGGER.report("Cleaning a Row..", error.LogType.INFO)

        newVertexRowCreated = 0
        newVertexRowsCreated = 0
        vertexRowList = []
        edgeRow = []

        phDInstId = 0
        apInstId = 0

        phDInstFound = 0
        apInstFound = 0

        phDInst: institution.Institution = None
        apInst: institution.Institution = None


        # find Ph.D institution

        for index in argRowIterator:

            if(util.areSameStrings(argTargetRow[index], 'PhD')):   

                instInfo = institution.InstInfo()

                if(not instInfo.addRegionToInstInfo(argTargetRow[index+2])): #better version
                    continue

                if(instInfo.isInvalid()):
                    continue
                
                instInfo.name = util.appendIfNotIn(instInfo.name, status.STATTRACKER.statInstNameList)
                instInfo.country = util.appendIfNotIn(instInfo.country, status.STATTRACKER.statCountryList)
                instInfo.region = util.appendIfNotIn(instInfo.region, status.STATTRACKER.statRegionList)

                vertexRow = []

                phDInstId = self.analyzer.getInstIdFor(instInfo)
                instInfo.addInstIdToInstInfo(phDInstId)

                newVertexRowCreated = self.__addToLocalInstIdList(phDInstId)

                phDInst = self.analyzer.getInstitution(instInfo, self.field)
                    
                newVertexRowsCreated = newVertexRowsCreated or newVertexRowCreated

                if(1 == newVertexRowCreated):

                    phDInst.flushInfoToList(vertexRow)
                    vertexRowList.append(vertexRow)

                phDInstFound = 1

                break

                
        # find Assistant Prof institution

        argRowIterator.changeTargetAndReset('Job')

        for index in argRowIterator:
            if(util.areSameStrings(argTargetRow[index], 'Assistant Professor')):

                instInfo = institution.InstInfo()

                if(not instInfo.addRegionToInstInfo(argTargetRow[index+1])):
                    continue

                if(instInfo.isInvalid()):
                    continue

                instInfo.name = util.appendIfNotIn(instInfo.name, status.STATTRACKER.statInstNameList)
                instInfo.country = util.appendIfNotIn(instInfo.country, status.STATTRACKER.statCountryList)
                instInfo.region = util.appendIfNotIn(instInfo.region, status.STATTRACKER.statRegionList)

                vertexRow = []
                
                apInstId = self.analyzer.getInstIdFor(instInfo)
                instInfo.addInstIdToInstInfo(apInstId)

                newVertexRowCreated = self.__addToLocalInstIdList(apInstId)

                apInst = self.analyzer.getInstitution(instInfo, self.field)

                newVertexRowsCreated = newVertexRowsCreated or newVertexRowCreated

                if(newVertexRowCreated):
                    apInst.flushInfoToList(vertexRow)
                    vertexRowList.append(vertexRow)

                apInstFound = 1
                
                break

        if(not (phDInstFound and apInstFound)):

            if(phDInstFound):
                status.STATTRACKER.statNumRowPhDWithNoAP[self.field] += 1
            elif(apInstFound):
                status.STATTRACKER.statNumRowAPWithNoPhD[self.field] += 1
            else:
                status.STATTRACKER.statNumRowWithNoPhDAndAP[self.field] += 1
                
            argRowIterator.changeTargetAndReset('Degree')
            error.LOGGER.report("This Row does not Contain PhD or AP Info", error.LogType.WARNING)

            status.STATTRACKER.statTotalNumRowCleaned += 1
            return 0
    
        
        currentRank = argTargetRow[argRowIterator.findFirstIndex("Current Rank", "APPROX")]
        gender = util.genderToStr(util.strToGender(\
            argTargetRow[argRowIterator.findFirstIndex("Sex", "APPROX")]))
        
        edgeRow.append(phDInst.instId)
        edgeRow.append(apInst.instId)
        edgeRow.append(gender)

        #add alumnus info
        gender = util.strToGender(gender)
        alumnusInfo = institution.AlumnusInfo(self.field, currentRank, gender, \
                                             phDInst.instId, phDInst.name, apInst.instId, apInst.name)
        phDInst.addAlumnusAt(alumnusInfo)


        if(1 == newVertexRowsCreated):
            for vertexRow in vertexRowList:
                self.__vertexList.loc[len(self.__vertexList)] = vertexRow
            
        self.__edgeList.loc[len(self.__edgeList)] = edgeRow

        argRowIterator.changeTargetAndReset('Degree')

        error.LOGGER.report("Successfully Cleaned a Row!", error.LogType.INFO)

        status.STATTRACKER.statTotalNumRowCleaned += 1
        return 1

    @error.callStackRoutine
    def __sortEdgeList(self):
        self.__edgeList.sort_values(by=['# u', '# v'], inplace=True)

    @error.callStackRoutine
    def __sortVertexList(self):
        self.__vertexList.sort_values(by = ['# u'], inplace = True)

    @error.callStackRoutine
    def calcRank(self):

        returnDict = {}
        
        for rankType in self.analyzer.getRankTypeList():
            if(institution.RankType.SPRANK == rankType):

                error.LOGGER.report("Calculating SpringRank, which is approximation of MVR Rank", error.LogType.INFO)
                error.LOGGER.report("Copyright (c) 2017 Caterina De Bacco and Daniel B Larremore", error.LogType.INFO)

                adjMat = np.zeros((len(self.__localInstIdList), len(self.__localInstIdList)), dtype = int)

                id2IndexMap = {}
                index2IdMap = {}

                targetRow = None

                localInstIdListSorted = sorted(self.__localInstIdList)

                for i in range(len(localInstIdListSorted)):
                    id2IndexMap[localInstIdListSorted[i]] = i
                    index2IdMap[i] = localInstIdListSorted[i]

                for numRow in range(len(self.__edgeList.index)):
                    targetRow = self.__edgeList.iloc[numRow]
                    adjMat[id2IndexMap[targetRow[0]], id2IndexMap[targetRow[1]]] += 1

            
                spRankList = sp.get_ranks(adjMat)
                inverseTemp = sp.get_inverse_temperature(adjMat, spRankList)

                rankList = util.getRankBasedOn(spRankList)
                rankList = util.getRidOfTie(rankList)

                for index in range(len(rankList)):
                    instId = index2IdMap[index]

                    self.analyzer.getExistingInstitution(instId).setRankAt(self.field, rankList[index], institution.RankType.SPRANK)

                error.LOGGER.report("Sucessfully Calculated SpringRank!", error.LogType.INFO)

                returnDict[rankType] = inverseTemp
            
            elif(institution.RankType.JRANK == rankType):

                targetDir = '../dataset/jrank'
                targetFileName = '_'.join(['jrank', self.field]) +  util.fileExtToStr(util.FileExt.CSV)

                if(False == os.path.isdir(targetDir)):
                    returnDict[rankType] = None
                    error.LOGGER.report(" ".join(["No Directory Named", targetDir]), error.LogType.CRITICAL)

                if(targetFileName not in os.listdir(targetDir)):
                    returnDict[rankType] = None
                    continue

                targetDf = util.readFileFor("/".join([targetDir, targetFileName]), [util.FileExt.CSV])
                rowIterator = util.rowIterator(targetDf.columns,'Inst')

                instNameList = []
                rankList = []

                for numRow in range(len(targetDf.index)):

                    targetRow = targetDf.iloc[numRow]

                    instName = targetRow[rowIterator.findFirstIndex('Inst', 'APPROX')]
                    instRank = targetRow[rowIterator.findFirstIndex('Rank', 'APPROX')]

                    instNameList.append(instName)
                    rankList.append(instRank)

                rankList = util.getRidOfTie(rankList)

                for instNum in range(len(instNameList)):
                    instName = instNameList[instNum]
                    rank = rankList[instNum]

                    inst = self.analyzer.getExistingInstitutionByName(instName)

                    if(None!=inst):
                        inst.setRankAt(self.field, rank, institution.RankType.JRANK)

                returnDict[rankType] = 1

        self.flags.raiseMVRRankCalculated()

        return returnDict

    @error.callStackRoutine
    def calcGiniCoeff(self):
        
        error.LOGGER.report("Calculating Gini Coefficient on # of Alumni", error.LogType.INFO)
        error.LOGGER.report("Result of This Method Can be Unreliable", error.LogType.WARNING)

        instDict = self.analyzer.getInstDict()
        numAlumniList = []

        for institution in util.getValuesListFromDict(instDict):
            if(institution.queryField(self.field)):
                numAlumniList.append(institution.getTotalNumAlumniAt(self.field))

        returnValue = util.calGiniCoeff(numAlumniList, "Faculty Production", self.field)

        self.flags.raiseGiniCoeffCalculated()

        error.LOGGER.report("Successfully Calculated Gini Coefficient on # of Alumni!", error.LogType.INFO)

        return returnValue

    """
    @error.callStackRoutine
    def calcMVRRank(self):
        
        error.LOGGER.report("Calculating MVR Rank", error.LogType.INFO)

        returnValue = self.calcSpringRank()

        self.flags.raiseMVRRankCalculated()

        error.LOGGER.report("Sucessfully Calculated MVR Rank!", error.LogType.INFO)
        return returnValue

    """
    
    @error.callStackRoutine
    def calcAvgMVRMoveBasedOnGender(self, argGender: util.Gender):

        if(not self.flags.ifMVRRankCalculated()):
            error.LOGGER.report("Attempt denied. MVR ranks are not pre-calculated.", error.LogType.ERROR)
            return 0
        
        if(not self.flags.ifMVRRankMoveCalculated()):
            error.LOGGER.report("Attempt denied. MVR rank movements are not pre-calculated.", error.LogType.ERROR)
            return 0

        if(argGender not in util.Gender):
            error.LOGGER.report("Invalid Gender", error.LogType.WARNING)
            return 0

        instDict = self.analyzer.getInstDict()
        rankMovementList = []

        for inst in util.getValuesListFromDict(instDict):
            
            department = inst.getFieldIfExists(self.field)

            if(None != department):
                if(argGender in department.alumniDictWithGenderKey):
                    for alumnus in department.alumniDictWithGenderKey[argGender]:

                        rankMovementList.append(alumnus.getRankMove(institution.RankType.SPRANK))


        return util.getMean(rankMovementList)
    
    @error.callStackRoutine
    def calcAvgMVRMoveForRange(self, argPercentLow: int, argPercentHigh: int):

        if(not self.flags.ifMVRRankCalculated()):
            error.LOGGER.report("Attempt denied. MVR ranks are not pre-calculated.", error.LogType.ERROR)
            return 0
        
        if(not self.flags.ifMVRRankMoveCalculated()):
            error.LOGGER.report("Attempt denied. MVR rank movements are not pre-calculated.", error.LogType.ERROR)
            return 0

        if(argPercentLow<=0):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(argPercentLow>100):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(int(argPercentLow) != argPercentLow):
            error.LOGGER.report("Invalid percentage. Floating point number not allowed.", error.LogType.WARNING)
        else:
            pass

        if(argPercentHigh<=0):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(argPercentHigh>100):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(int(argPercentHigh) != argPercentHigh):
            error.LOGGER.report("Invalid percentage. Floating point number not allowed.", error.LogType.WARNING)
        else:
            pass

        if(argPercentLow > argPercentHigh):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)

        error.LOGGER.report("This function (incomplete) doees not calculate based on percentage.", error.LogType)

        rankMovementList = []

        rankList = []
        instList = []

        instDict = self.analyzer.getInstDict()
        
        for inst in util.getValuesListFromDict(instDict):
            department = inst.getFieldIfExists(self.field)

            if(None != department):    #institution has self.field field
                instList.append(inst)
                rankList.append(inst.getRankAt(self.field, institution.RankType.SPRANK))

        rankLowerBound = math.ceil(float(len(rankList) * argPercentLow / 100))
        rankUpperBound = math.floor(float(len(rankList)* argPercentHigh / 100))

        
        instListSorted = util.sortObjectBasedOn(instList, rankList)
        
        for inst in instListSorted[rankLowerBound:rankUpperBound+1]:
            department = inst.getFieldIfExists(self.field)

            if(None != department):    #institution has self.field field
                for alumni in util.getValuesListFromDict(department.alumniDict):
                    for alumnus in alumni:

                        rankMovementList.append(alumnus.getRankMove(institution.RankType.SPRANK))

        return util.getMean(rankMovementList)
    
    @error.callStackRoutine
    def calcMVRMoveForAllAlumni(self):

        if(not self.flags.ifMVRRankCalculated()):
            return 0
        
        instDict = self.analyzer.getInstDict()
        
        for inst in util.getValuesListFromDict(instDict):
            department = inst.getFieldIfExists(self.field)

            if(None != department):
                for alumnus in department.getTotalAlumniList():

                    for rankType in self.analyzer.getRankTypeList():
                        phDInstRank = instDict[alumnus.phDInstId].getRankAt(self.field, rankType)
                        apInstRank = instDict[alumnus.apInstId].getRankAt(self.field, rankType)

                        if(None!=phDInstRank and None!=apInstRank):
                            alumnus.setRankMove(apInstRank - phDInstRank, rankType)
                        else:
                            error.LOGGER.report("None-value ranks are ignored.", error.LogType.DEBUG)

        self.flags.raiseMVRRankMoveCalculated()

        return 1
    

    @error.callStackRoutine
    def plotRankMove(self, argPercentLow: int, argPercentHigh: int):

        if(argPercentLow<0):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(argPercentLow>100):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(int(argPercentLow) != argPercentLow):
            error.LOGGER.report("Invalid percentage. Floating point number not allowed.", error.LogType.WARNING)
        else:
            pass

        if(argPercentHigh<0):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(argPercentHigh>100):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(int(argPercentHigh) != argPercentHigh):
            error.LOGGER.report("Invalid percentage. Floating point number not allowed.", error.LogType.WARNING)
        else:
            pass

        if(argPercentLow > argPercentHigh):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)

        instDict = self.analyzer.getInstDict()
        numTotalInst = self.getTotalNumInstInField()

        rankLowerBound = math.ceil(float(numTotalInst * argPercentLow / 100))
        rankUpperBound = math.floor(float(numTotalInst* argPercentHigh / 100))

        maleMoveList = []
        femaleMoveList = []
        totalMoveList = []

        for inst in util.getValuesListFromDict(instDict):
            
            department = inst.getFieldIfExists(self.field)

            if(None != department):

                for alumnus in department.getTotalAlumniList():
                    data = alumnus.getRankMove(institution.RankType.SPRANK) / numTotalInst

                    phDInstRank = instDict[alumnus.phDInstId].getRankAt(self.field, institution.RankType.SPRANK)

                    if(rankLowerBound <= phDInstRank <= rankUpperBound):
                        if(util.Gender.MALE == alumnus.getGender()):
                            maleMoveList.append(data)
                        elif(util.Gender.FEMALE == alumnus.getGender()):
                            femaleMoveList.append(data)

                        totalMoveList.append(data)

        maleMoveList = sorted(maleMoveList)
        femaleMoveList = sorted(femaleMoveList)
        totalMoveList = sorted(totalMoveList)

        font = {'family': 'serif', 'size': 9}
        titleStr = "Relative SpringRank Change Distribution " + "(Field: " + self.field + ")"

        binList = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, \
                   0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        plt.rc('font', **font)

        plt.title(titleStr)
        plt.xlabel("Relative change in rank from PhD")
        plt.ylabel("Number of Alumni")

        plt.xlim(-1, 1)

        if('Physics' == self.field):
            plt.ylim(0, 100)
        elif('Computer Science' == self.field):
            plt.ylim(0, 200)

        #plt.xlim(np.float32(0), np.float32(100))
        #plt.ylim(np.float32(0), np.float32(100))

        #plt.plot(maleXCoList, maleYCoList, color = 'blue')
        #plt.plot(femaleXCoList, femaleYCoList, linestyle = "--", color = 'blue')

        plt.hist(maleMoveList, bins = binList, label='Male')
        plt.hist(femaleMoveList, bins = binList, color= 'skyblue', label='Female')

        plt.legend()

        #plt.plot(xCoList, maleYCoList, color = 'blue')
        #plt.plot(xCoList, femaleYCoList, linestyle = '--', color = 'blue')

        #plt.plot(xCoList, maleYCoListNormed, color = 'blue')
        #plt.plot(xCoList, femaleYCoListNormed, linestyle = '--', color = 'blue')

        figPath = util.getPlotPath("RankMove", "Hist", self.field, argPercentLow, argPercentHigh)

        if(0 == argPercentLow and 100 == argPercentHigh):
            figPath = util.getPlotPath("RankMove", "Hist", self.field, "Total")


        plt.savefig(figPath)
        plt.clf()

    @error.callStackRoutine
    def plotGenderRatio(self):

        if(not self.flags.ifMVRRankCalculated()):
            error.LOGGER.report("Attempt denied. MVR ranks are not pre-calculated.", error.LogType.ERROR)
            return 0

        instDict = self.analyzer.getInstDict()
        numTotalInst = self.getTotalNumInstInField()
        genderRatioDict = {}

        xCoList = []
        yCoList = []

        for inst in util.getValuesListFromDict(instDict):
            
            department = inst.getFieldIfExists(self.field)

            if(None != department):

                numMale = 0
                numFemale = 0
                rank = department.MVRRank

                for alumnus in department.getTotalAlumniList():
                    #data = alumnus.getRankMove(institution.RankType.SPRANK) / numTotalInst

                    #phDInstRank = instDict[alumnus.phDInstId].getRankAt(self.field, institution.RankType.SPRANK)

                    if(util.Gender.MALE == alumnus.gender):
                        numMale+=1
                    elif(util.Gender.FEMALE == alumnus.gender):
                        numFemale+=1
                    else:
                        error.LOGGER.report("Invalid Gender.", error.LogType.CRITICAL)
                        return 0
                
                if(0 != numMale + numFemale):
                    genderRatioDict[rank] = np.float32(numFemale / (numMale + numFemale))


                else:
                    genderRatioDict[rank] = np.float32(-0.1) 

            
        for rank in range(1, len(genderRatioDict)+1):
            xCoList.append(rank)
            yCoList.append(genderRatioDict[rank])

        font = {'family': 'serif', 'size': 9}
        titleStr = "Ratio of Female Alumni " + "(Field: " + self.field + ")"

        plt.rc('font', **font)

        plt.title(titleStr)
        plt.xlabel("SpringRank")
        plt.ylabel("Ratio of Female Alumni")

        plt.plot(xCoList, yCoList, color='blue')

        plt.legend()

        figPath = util.getPlotPath("GenderRatio", "Plot", self.field)

        plt.savefig(figPath)
        plt.clf()

        return 1
    
    @error.callStackRoutine
    def plotNonKRFac(self):

        if(not self.flags.ifMVRRankCalculated()):
            error.LOGGER.report("Attempt denied. MVR ranks are not pre-calculated.", error.LogType.ERROR)

            return 0

        instDict = self.analyzer.getInstDict()
        rankTypeList = self.analyzer.getRankTypeList()

        nonKRProfDict = {}
        xCoDict = {}
        yCoDict = {}

        for rankType in rankTypeList:
            nonKRProfDict[rankType] = {}
            xCoDict[rankType] = []
            yCoDict[rankType] = []

        for inst in util.getValuesListFromDict(instDict):
            
            department = inst.getFieldIfExists(self.field)

            for rankType in rankTypeList:

                if(None != department):

                    phDInstRank = inst.getRankAt(self.field, rankType)

                    if(None != phDInstRank and phDInstRank not in nonKRProfDict[rankType]):
                        nonKRProfDict[rankType][phDInstRank] = 0
                    
                    isNonKRPhD = inst.isNonKRInst()

                    for alumnus in department.getTotalAlumniList():

                        apInstRank = instDict[alumnus.apInstId].getRankAt(self.field, rankType)

                        if(None == apInstRank):
                            continue

                        if(apInstRank in nonKRProfDict[rankType]):
                            nonKRProfDict[rankType][apInstRank] += isNonKRPhD
                        else:
                            nonKRProfDict[rankType][apInstRank] = isNonKRPhD




        font = {'family': 'serif', 'size': 9}
        titleStr = "Number of Faculty with Non-KR PhD " + "(Field: " + self.field + ")"

        for rankType in rankTypeList:

            for rank in sorted(util.getKeyListFromDict(nonKRProfDict[rankType])):

                xCoDict[rankType].append(rank)
                yCoDict[rankType].append(nonKRProfDict[rankType][rank])
            
            plt.rc('font', **font)

            plt.title(titleStr)
            plt.xlabel(rankType.toStr("camelcase"))
            plt.ylabel("Number of Faculty with Non-KR PhD")

            plt.plot(xCoDict[rankType], yCoDict[rankType], color='blue')

            plt.legend()

            figPath = util.getPlotPath("NumNonKRFac", "Plot", self.field, rankType.toStr("abbrev"))

            plt.savefig(figPath)
            plt.clf()

        return 1
    
    @error.callStackRoutine
    def plotNonKRFacRatio(self):

        if(not self.flags.ifMVRRankCalculated()):
            error.LOGGER.report("Attempt denied. MVR ranks are not pre-calculated.", error.LogType.ERROR)

            return 0

        instDict = self.analyzer.getInstDict()
        rankTypeList = self.analyzer.getRankTypeList()

        nonKRProfDict = {}
        profDict = {}
        xCoDict = {}
        yCoDict = {}

        for rankType in rankTypeList:
            nonKRProfDict[rankType] = {}
            profDict[rankType] = {}
            xCoDict[rankType] = []
            yCoDict[rankType] = []

        for inst in util.getValuesListFromDict(instDict):
            
            department = inst.getFieldIfExists(self.field)

            for rankType in rankTypeList:

                if(None != department):

                    phDInstRank = inst.getRankAt(self.field, rankType)

                    if(None != phDInstRank and phDInstRank not in nonKRProfDict[rankType]):
                        nonKRProfDict[rankType][phDInstRank] = 0

                    if(None != phDInstRank and phDInstRank not in profDict[rankType]):
                        profDict[rankType][phDInstRank] = 0
                    
                    isNonKRPhD = inst.isNonKRInst()

                    for alumnus in department.getTotalAlumniList():

                        apInstRank = instDict[alumnus.apInstId].getRankAt(self.field, rankType)

                        if(None == apInstRank):
                            continue

                        if(apInstRank in nonKRProfDict[rankType]):
                            nonKRProfDict[rankType][apInstRank] += isNonKRPhD
                        else:
                            nonKRProfDict[rankType][apInstRank] = isNonKRPhD
                        
                        if(apInstRank in profDict[rankType]):
                            profDict[rankType][apInstRank] += 1

                        else:
                            profDict[rankType][apInstRank] = 1

        font = {'family': 'serif', 'size': 9}
        titleStr = "Ratio of Faculty with Non-KR PhD " + "(Field: " + self.field + ")"

        for rankType in rankTypeList:

            for rank in sorted(util.getKeyListFromDict(nonKRProfDict[rankType])):

                xCoDict[rankType].append(rank)

                if(0 != profDict[rankType][rank]):
                    ratio = (np.float32) (nonKRProfDict[rankType][rank]/ profDict[rankType][rank])
                else:
                    ratio = -1

                yCoDict[rankType].append(ratio)
            
            plt.rc('font', **font)

            plt.title(titleStr)
            plt.xlabel(rankType.toStr("camelcase"))
            plt.ylabel("Ratio of Faculty with Non-KR PhD")

            plt.plot(xCoDict[rankType], yCoDict[rankType], color='blue')

            plt.legend()

            figPath = util.getPlotPath("RatioNonKRFac", "Plot", self.field, rankType.toStr("abbrev"))

            plt.savefig(figPath)
            plt.clf()

        return 1

    @error.callStackRoutine
    def __exportXLSX(self):

        self.__sortVertexList()
        self.__sortEdgeList()

        #vertexFilename = ''.join(['_'.join(['vertex', self.field, currentDateStr]), '.xlsx'])
        #edgeFilename = ''.join(['_'.join(['edge', self.field, currentDateStr]), '.xlsx'])

        #vertexFilePath = os.path.join('../dataset/cleaned/', vertexFilename)
        #edgeFilePath = os.path.join('../dataset/cleaned/', edgeFilename)

        vertexFilePath = util.getCleanedFilePath("Vertex", self.field, util.FileExt.XLSX)
        edgeFilePath = util.getCleanedFilePath("Edge", self.field, util.FileExt.XLSX)
        

        self.__vertexList.to_excel(vertexFilePath, index = False, engine = 'openpyxl')
        self.__edgeList.to_excel(edgeFilePath, index= False, engine = 'openpyxl')
    

    @error.callStackRoutine
    def __exportCSV(self):

        self.__sortVertexList()
        self.__sortEdgeList()

        #vertexFilename = ''.join(['_'.join(['vertex', self.field, currentDateStr]), '.csv'])
        #edgeFilename = ''.join(['_'.join(['edge', self.field, currentDateStr]), '.csv'])

        #vertexFilePath = os.path.join('../dataset/cleaned/', vertexFilename)
        #edgeFilePath = os.path.join('../dataset/cleaned/', edgeFilename)

        vertexFilePath = util.getCleanedFilePath("Vertex", self.field, util.FileExt.CSV)
        edgeFilePath = util.getCleanedFilePath("Edge", self.field, util.FileExt.CSV)

        self.__vertexList.to_csv(vertexFilePath, index = False, sep = '\t')
        self.__edgeList.to_csv(edgeFilePath, index = False, sep = '\t')

    @error.callStackRoutine
    def exportVertexAndEdgeListAs(self, argFileExtension: util.FileExt):

        if(util.FileExt.XLSX == argFileExtension):
            self.__exportXLSX()
        elif(util.FileExt.CSV == argFileExtension):
            self.__exportCSV()
        else:
            error.LOGGER.report(": ".join(["Invalid File Extension", util.fileExtToStr(argFileExtension)], error.LogType.ERROR))

if(__name__ == '__main__'):

    error.LOGGER.report("This Module is Not for Main Function", error.LogType.CRITICAL)

    
