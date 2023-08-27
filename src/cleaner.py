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
import setting
import career
import status
import math
import os
import waiter

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad

"""
class Cleaner

For every single .xlsx file in dataset_dirty directory, 
clean them and make two output .csv files (vertex list and edge list) in dataset_cleaned directory
Cleaner is allocated 'per field', by class analyzer.Analyzer. 
Cleaner is also responsible for any calculation such as Gini Coefficient or MVR Ranking, etc.

"""
class Cleaner():

    # ================================================================================== #
    # 0. Constructor
    # ================================================================================== #

    @error.callStackRoutine
    def __init__(self, argAnalyzer, argField, argKorea):

        self.analyzer= argAnalyzer

        self.field = argField

        self.__vertexList = pd.DataFrame(columns = ['# u', 'Institution', 'Region', 'Country'])

        self.__edgeListDict = {}

        self.targetDegList = setting.PARAM["Basic"]["targetDeg"]

        for index in range(len(self.targetDegList)):
            if(index >= len(self.targetDegList) - 1):
                break

            srcDegType = self.targetDegList[index]
            dstDegType = self.targetDegList[index+1]

            self.__edgeListDict[(srcDegType, dstDegType)] = pd.DataFrame(columns = ['# u', '# v', 'gender', 'pid'])
        
        self.__localInstIdList = []

        self.korea = argKorea

        self.scholarDict = {}

        self.flags = util.Flag(['RankSet'])

    # ================================================================================== #
    # 1. Methods for Management
    # ================================================================================== #

    # ====================================================== #
    # 1.a. Methods for Managing Institutions in Field
    # ====================================================== #

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
    
    # ====================================================== #
    # 1.b. Methods for Cleaning Data
    # ====================================================== #
        
    @error.callStackRoutine
    def __cleanStep(self, argTargetRow, argRowIterator: util.rowIterator, argDegType: career.DegreeType):

        hit = 0
        step = None
        newVertexRowCreated = 0
        vertexRow = []

        offset = 2

        key = argDegType.toStr("search")

        if(argDegType in [career.DegreeType.PDOC, career.DegreeType.AP]):
            argRowIterator.changeTargetAndReset('Job')
            offset = 1

        for index in argRowIterator:

            if(util.areSameStrings(argTargetRow[index], key)):   

                instInfo = institution.InstInfo()

                if(not instInfo.addRegionToInstInfo(argTargetRow[index+offset])): #better version
                    continue

                if(instInfo.isInvalid()):
                    continue
                
                instInfo.name = util.appendIfNotIn(instInfo.name, status.STATTRACKER.statInstNameList)
                instInfo.country = util.appendIfNotIn(instInfo.country, status.STATTRACKER.statCountryList)
                instInfo.region = util.appendIfNotIn(instInfo.region, status.STATTRACKER.statRegionList)

                phDInstId = self.analyzer.getInstIdFor(instInfo)
                instInfo.addInstIdToInstInfo(phDInstId)

                newVertexRowCreated = self.__addToLocalInstIdList(phDInstId)

                phDInst = self.analyzer.getInstitution(instInfo, self.field)

                step = career.CareerStep(phDInst, argDegType)

                if(newVertexRowCreated):
                    phDInst.flushInfoToList(vertexRow)
                    
                hit = 1

                break
        
        argRowIterator.changeTargetAndReset('Degree')

        return (hit, step, newVertexRowCreated, vertexRow) 

    @error.callStackRoutine
    def cleanRow(self, argTargetRow, argRowIterator):
        error.LOGGER.report("Cleaning a Row..", error.LogType.INFO)

        newVertexRowCreated = 0
        vertexRowList = []
        edgeRow = []

        hitDict = {}
        careerStepDict = {}

        # Step 1: Inspect the target row and gather informations.

        for degType in self.targetDegList:

            (hit, step, newVertexRowCreated, vertexRow) = \
                self.__cleanStep(argTargetRow, argRowIterator, degType)
            
            hitDict[degType] = hit

            if(hit):
                careerStepDict[degType] = step
            
            if(newVertexRowCreated):
                vertexRowList.append(vertexRow)
        
        # Step 2: Construct the career of the alumnus.

        currentRank = argTargetRow[argRowIterator.findFirstIndex("Current Rank", "APPROX")]
        gender = util.genderToStr(util.strToGender(\
            argTargetRow[argRowIterator.findFirstIndex("Sex", "APPROX")]))
        
        pid = self.analyzer.getNextPID()

        alumnus = career.Alumni(self.field, currentRank, util.strToGender(gender), pid)

        self.scholarDict[pid] = alumnus

        for step in util.getValuesListFromDict(careerStepDict):
            alumnus.append(step)

        # Step 3: Maintain vertex list and edge list.

        for vertexRow in vertexRowList:
            self.__vertexList.loc[len(self.__vertexList)] = vertexRow

        for index in range(len(self.targetDegList)):
            if(index >= len(self.targetDegList) - 1):
                break

            srcDegType = self.targetDegList[index]
            dstDegType = self.targetDegList[index+1]

            if(hitDict[srcDegType] and hitDict[dstDegType]):
                    
                targetEdgeList = self.__edgeListDict[(srcDegType, dstDegType)]
                edgeRow = []

                srcInstId = careerStepDict[srcDegType].inst.instId
                dstInstId = careerStepDict[dstDegType].inst.instId
                
                edgeRow.append(srcInstId)
                edgeRow.append(dstInstId)
                edgeRow.append(gender)
                edgeRow.append(pid)

                targetEdgeList.loc[len(targetEdgeList)] = edgeRow

            else:
                continue

        argRowIterator.changeTargetAndReset('Degree')

        error.LOGGER.report("Successfully Cleaned a Row!", error.LogType.INFO)

        status.STATTRACKER.statTotalNumRowCleaned += 1

        return 1
    

    # ====================================================== #
    # 1.c. Others
    # ====================================================== #
    
    @error.callStackRoutine
    def isClosedSys(self):
        return "CLOSED" == setting.PARAM["Basic"]["networkType"]

    # ================================================================================== #
    # 2. Methods for Analysis
    # ================================================================================== #

    # ====================================================== #
    # 2.a. Methods for Setting Ranks
    # ====================================================== #

    @error.callStackRoutine
    def __calcSpRankForStep(self, argDegTuple, argDegForRankSet, argFilePathForRankSet):

        srcDegType = argDegTuple[0]
        dstDegType = argDegTuple[1]

        if(srcDegType not in self.targetDegList or srcDegType not in self.targetDegList):
            error.LOGGER.report("Invalid argDegType.", error.LogType.WARNING)
            return None

        targetEdgeList = self.__edgeListDict[(srcDegType, dstDegType)]
        targetInstIdList = []

        # 1. Gather targetInstIdList

        if(None!=argFilePathForRankSet):
            targetDf = util.readFileFor(argFilePathForRankSet, [util.FileExt.XLSX])

            targetInstList = targetDf['Inst'].tolist()

            for instId in self.__localInstIdList:
            
                inst = self.analyzer.getExistingInstitution(instId)

                if(None == inst):
                    continue

                instInfoList = []

                inst.flushInfoToList(instInfoList)

                instInfoStr = ', '.join(instInfoList[1:])

                if(util.ifStrMatchesAmong(instInfoStr, targetInstList)):
                    targetInstIdList.append(instId)

        else:
            targetInstIdList = self.__localInstIdList

        # 2. Make adjMat

        matSize = len(targetInstIdList)

        adjMat = np.zeros((matSize, matSize), dtype = int)

        id2IndexMap = {}
        index2IdMap = {}

        targetRow = None

        targetInstIdListSorted = sorted(targetInstIdList)

        for i in range(len(targetInstIdListSorted)):
            id2IndexMap[targetInstIdListSorted[i]] = i
            index2IdMap[i] = targetInstIdListSorted[i]

        for numRow in range(len(targetEdgeList.index)):
            targetRow = targetEdgeList.iloc[numRow]

            srcId = targetRow[0]
            dstId = targetRow[1]

            if(all(id in targetInstIdList for id in [srcId, dstId])):

                if(career.DegreeType.PHD == argDegForRankSet):
                    status.STATTRACKER.statTotalNumAlumniInScope[self.field] += 1

                adjMat[id2IndexMap[srcId], id2IndexMap[dstId]] += 1

        adjMat += 1     #adding 1 to all elements for precise anlaysis

        # 3. Get SpringRank Scores

        error.LOGGER.report("Calculating SpringRank.", error.LogType.INFO)
        error.LOGGER.report("Copyright (c) 2017 Caterina De Bacco and Daniel B Larremore", error.LogType.INFO)

        spScoreList = sp.get_ranks(adjMat)
        inverseTemp = sp.get_inverse_temperature(adjMat, spScoreList)

        error.LOGGER.report("Sucessfully Calculated SpringRank!", error.LogType.INFO)

        # 4. Convert Scores into Ranks

        rankList = util.getRankBasedOn(spScoreList)
        rankList = util.getRidOfTie(rankList)

        # 5. Record the Ranks


        for index in range(len(rankList)):
            instId = index2IdMap[index]
            rank = rankList[index]

            self.analyzer.getExistingInstitution(instId).setRankAt(self.field, rank, argDegForRankSet)

        return inverseTemp
    
    @error.callStackRoutine
    def setRanks(self):

        returnDict = {}

        for degTypeForRankSet in self.targetDegList:
            rankTypeDescription = setting.PARAM["RankType"][degTypeForRankSet]

            rankType = rankTypeDescription[0]

            match rankType:
                case "SpringRank":

                    degTuple = rankTypeDescription[1]
                    targetInstListPath = rankTypeDescription[2]

                    invTemp = self.__calcSpRankForStep(degTuple, degTypeForRankSet, targetInstListPath)

                    returnDict[degTypeForRankSet] = invTemp

                case "Custom":
                    rankFilePath = rankTypeDescription[1]

                    targetDf = pd.read_excel(rankFilePath, sheet_name=self.field)

                    instNameList = []
                    rankList = []

                    for numRow in range(len(targetDf.index)):

                        targetRow = targetDf.iloc[numRow]

                        instRank = targetRow[0]
                        instName = targetRow[1]

                        instNameList.append(instName)
                        rankList.append(instRank)

                    rankList = util.getRidOfTie(rankList)

                    for instNum in range(len(instNameList)):
                        instName = instNameList[instNum]
                        rank = rankList[instNum]

                        inst = self.analyzer.getExistingInstitutionByName(instName)

                        if(None!=inst):
                            inst.setRankAt(self.field, rank, degTypeForRankSet)

                    returnDict[degTypeForRankSet] = 1

                case _:
                    pass

        self.flags.raiseFlag('RankSet')

    # ====================================================== #
    # 2.b. Methods for Calculation of Metrics
    # ====================================================== #
    
    @error.callStackRoutine
    def calcAvgMVRMoveBasedOnGender(self, argGender: util.Gender, argDegTuple):
        rankMoveList = []

        if(not self.flags.ifRaised('RankSet')):
            error.LOGGER.report("Attempt denied. MVR ranks are not pre-calculated.", error.LogType.ERROR)
            return 0
        
        if(argGender not in util.Gender):
            error.LOGGER.report("Invalid Gender", error.LogType.WARNING)
            return 0
        
        for alumnus in util.getValuesListFromDict(self.scholarDict):
            if(argGender == alumnus.getGender()):

                rankMove = alumnus.getRankMove(argDegTuple)

                if(None!=rankMove):
                    rankMoveList.append(rankMove)

        return util.getMean(rankMoveList)
    
    @error.callStackRoutine
    def calcAvgMVRMoveForRange(self, argRangeTuple, argDegTuple):

        if(not self.flags.ifRankSet):
            error.LOGGER.report("Attempt denied. MVR ranks are not pre-calculated.", error.LogType.ERROR)
            return 0
        elif(not(isinstance(argRangeTuple, tuple) and isinstance(argDegTuple, tuple))):
            return 0
        
        percentLow = argRangeTuple[0]
        percentHigh = argRangeTuple[1]

        srcDegType = argDegTuple[0]
        
        if(percentLow<=0):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(percentLow>100):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(int(percentLow) != percentLow):
            error.LOGGER.report("Invalid percentage. Floating point number not allowed.", error.LogType.WARNING)
        else:
            pass

        if(percentHigh<=0):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(percentHigh>100):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(int(percentHigh) != percentHigh):
            error.LOGGER.report("Invalid percentage. Floating point number not allowed.", error.LogType.WARNING)
        else:
            pass

        if(percentLow > percentHigh):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)

        rankMoveList = []

        rankList = []
        instList = []

        instDict = self.analyzer.getInstDict()
        
        for inst in util.getValuesListFromDict(instDict):
            department = inst.getFieldIfExists(self.field)

            if(None != department):    #institution has self.field field

                rank = inst.getRankAt(self.field, srcDegType)

                if(None != rank):   #rank is valid
                    instList.append(inst)
                    rankList.append(rank)

        lowerBoundIndex = math.ceil(float(len(rankList) * percentLow / 100))
        upperBoundIndex = math.floor(float(len(rankList)* percentHigh / 100))

        lowerBoundIndex = 0 if(lowerBoundIndex < 0) else lowerBoundIndex
        upperBoundIndex = len(rankList)-1 if(upperBoundIndex > len(rankList)-1) else upperBoundIndex

        rankLowerBound = sorted(rankList)[lowerBoundIndex]
        rankUpperBound = sorted(rankList)[upperBoundIndex]

        for index in range(len(rankList)):

            rank = rankList[index]

            if(rankLowerBound <= rank < rankUpperBound):

                inst = instList[index]
                department = inst.getFieldIfExists(self.field)

                if(None != department):
                    for alumnus in department.getTotalAlumniListForDeg(srcDegType):

                        rankMove = alumnus.getRankMove(argDegTuple)

                        if(None!=rankMove):
                            rankMoveList.append(rankMove)

        return util.getMean(rankMoveList)
    
    # ====================================================== #
    # 2.c. Methods for Plotting
    # ====================================================== #

    @error.callStackRoutine
    def plotRankMove(self, argRangeTuple: tuple, argDegTuple: tuple):

        if(not self.flags.ifRaised('RankSet')):
            error.LOGGER.report("Attempt denied. MVR ranks are not pre-calculated.", error.LogType.ERROR)
            return 0
        elif(not(isinstance(argRangeTuple, tuple) and isinstance(argDegTuple, tuple))):
            return 0

        percentLow = argRangeTuple[0]
        percentHigh = argRangeTuple[1]

        srcDegType = argDegTuple[0]
        dstDegType = argDegTuple[1]

        if(percentLow<0):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(percentLow>100):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(int(percentLow) != percentLow):
            error.LOGGER.report("Invalid percentage. Floating point number not allowed.", error.LogType.WARNING)
        else:
            pass

        if(percentHigh<0):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(percentHigh>100):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)
        elif(int(percentHigh) != percentHigh):
            error.LOGGER.report("Invalid percentage. Floating point number not allowed.", error.LogType.WARNING)
        else:
            pass

        if(percentLow > percentHigh):
            error.LOGGER.report("Invalid percentage range.", error.LogType.WARNING)

        instDict = self.analyzer.getInstDict()

        rankList = []
        instList = []

        for inst in util.getValuesListFromDict(instDict):
            department = inst.getFieldIfExists(self.field)

            if(None != department):    #institution has self.field field

                rank = inst.getRankAt(self.field, srcDegType)

                if(None != rank):   #rank is valid
                    rankList.append(rank)
                    instList.append(inst)

        lowerBoundIndex = math.ceil(float(len(rankList) * percentLow / 100))
        upperBoundIndex = math.floor(float(len(rankList)* percentHigh / 100))

        lowerBoundIndex = 0 if(lowerBoundIndex < 0) else lowerBoundIndex
        upperBoundIndex = len(rankList)-1 if(upperBoundIndex > len(rankList)-1) else upperBoundIndex

        rankLowerBound = sorted(rankList)[lowerBoundIndex]
        rankUpperBound = sorted(rankList)[upperBoundIndex]

        maleMoveList = []
        femaleMoveList = []
        totalMoveList = []

        for inst in instList:
            
            department = inst.getFieldIfExists(self.field)

            for alumnus in department.getTotalAlumniListForDeg(srcDegType):

                rankMove = alumnus.getRankMove(argDegTuple)

                if(None == rankMove):
                    continue

                data = rankMove / len(rankList)

                srcInstRank = alumnus.getInstFor(srcDegType).getRankAt(self.field, srcDegType)

                if(rankLowerBound <= srcInstRank < rankUpperBound):
                    if(util.Gender.MALE == alumnus.getGender()):
                        maleMoveList.append(data)
                    elif(util.Gender.FEMALE == alumnus.getGender()):
                        femaleMoveList.append(data)

                    totalMoveList.append(data)

        maleMoveList = sorted(maleMoveList)
        femaleMoveList = sorted(femaleMoveList)
        totalMoveList = sorted(totalMoveList)

        font = {'family': 'Helvetica', 'size': 9}

        plt.rc('font', **font)
        plt.figure(figsize=(7,5), dpi=200)

        titleStr = f"Relative Rank Change {srcDegType.toStr('label')}_{dstDegType.toStr('label')} Distribution {percentLow}% - {percentHigh}% (Field: {self.field})"

        binList = [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, \
                   0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        plt.title(titleStr)
        plt.xlabel(f"Relative Change in Rank ({srcDegType.toStr('label')}_{dstDegType.toStr('label')})")
        plt.ylabel("Number of Alumni")

        plt.xlim(-1, 1)

        """
        if('Physics' == self.field):
            plt.ylim(0, 100)
        elif('Computer Science' == self.field):
            plt.ylim(0, 200)
        """    

        plt.hist(maleMoveList, bins = binList, label='Male')
        plt.hist(femaleMoveList, bins = binList, color= 'skyblue', label='Female')

        plt.legend()

        figPath = waiter.WAITER.getPlotPath(f"RankMove {srcDegType.toStr('label')}_{dstDegType.toStr('label')}", "Hist", self.field, percentLow, percentHigh)

        if(0 == percentLow and 100 == percentHigh):
            figPath = waiter.WAITER.getPlotPath(f"RankMove {srcDegType.toStr('label')}_{dstDegType.toStr('label')}", "Hist", self.field, "Total")


        plt.savefig(figPath)
        plt.clf()

    @error.callStackRoutine
    def plotGenderRatio(self):

        if(not self.flags.ifRaised('RankSet')):
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
                rank = department.spRank

                for alumnus in department.getTotalAlumniListForDeg(career.DegreeType.PHD):

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

        figPath = waiter.WAITER.getPlotPath("GenderRatio", "Plot", self.field)

        plt.savefig(figPath)
        plt.clf()

        return 1

    @error.callStackRoutine
    def plotNonKR(self, argDegTuple: tuple, argKROnly, argSizeOfCluster):

        if(not isinstance(argDegTuple, tuple)):
            error.LOGGER.report("argDegTuple should be a tuple.", error.LogType.ERROR)
            return 0
        elif(2 != len(argDegTuple)):
            error.LOGGER.report("argDegTuple should have 2 elements.", error.LogType.ERROR)
            return 0
        elif(any(not isinstance(degType, career.DegreeType) for degType in argDegTuple)):
            error.LOGGER.report("argDegTuple should have career.DegreeType instances as its elements.", error.LogType.ERROR)
            return 0

        srcDegType = argDegTuple[0]
        dstDegType = argDegTuple[1]

        instDict = self.analyzer.getInstDict()

        rankList = []

        nonKRDict = {}
        krDict = {}

        for inst in util.getValuesListFromDict(instDict):
            
            department = inst.getFieldIfExists(self.field)

            if(argKROnly and inst.isNonKRInst()):
                continue

            if(None != department):

                dstRank = inst.getRankAt(self.field, dstDegType)

                if(None != dstRank):

                    rankList.append(dstRank)
            
        rankList.sort()

        for index in range(len(rankList)):
            nonKRDict[index+1] = 0
            krDict[index+1] = 0

        for alumnus in util.getValuesListFromDict(self.scholarDict):
            
            srcInst = alumnus.getInstFor(srcDegType)
            dstInst = alumnus.getInstFor(dstDegType)

            if(None == srcInst or None == dstInst):
                continue
            elif(dstInst.getRankAt(self.field, dstDegType) not in rankList):
                continue

            if(srcInst.isNonKRInst()):
                nonKRDict[rankList.index(dstInst.getRankAt(self.field, dstDegType)) + 1] += 1
            else:
                krDict[rankList.index(dstInst.getRankAt(self.field, dstDegType)) + 1] += 1

        rankIndexList = [rankList.index(rank) + 1 for rank in rankList]
        nonKRList = [nonKRDict[rankList.index(rank) + 1] for rank in rankList]
        krList = [krDict[rankList.index(rank) + 1] for rank in rankList]


        util.listsToDataFrame(['Rank', 'NonKR', 'KR'], [rankList, nonKRList, krList]).to_excel("../nonkr_global.xlsx")

        #rankIndexList = rankIndexList[::-1]
        #nonKRList = nonKRList[::-1]
        #krList = krList[::-1]

        if(1 == argSizeOfCluster):
            xCoList = rankIndexList
            nonKRYCoList = nonKRList
            krYCoList = krList

        else:

            xCoList = []
            nonKRYCoList = []
            krYCoList = []

            index = 0
            clusterId = 1
            sizeOfCluster = argSizeOfCluster

            while index < len(rankIndexList):

                nonKRElementList = []
                krElementList = []

                while(sizeOfCluster > len(nonKRElementList)):
                    nonKRElementList.append(nonKRList[index])
                    krElementList.append(krList[index])

                    index += 1

                    if(index >= len(rankIndexList)):
                        break

                xCoList.append(clusterId)
                nonKRYCoList.append(sum(nonKRElementList))
                krYCoList.append(sum(krElementList))

                clusterId += 1

        font = {'family': 'Helvetica', 'size': 9}

        plt.rc('font', **font)
        plt.figure(figsize=(7,5), dpi=200)

        titleStr = f"Number of {dstDegType.toStr('label')} with Non-KR {srcDegType.toStr('label')}"

        plt.title(titleStr)
        plt.xlabel(f"Number of {dstDegType.toStr('label')} with Non-KR {srcDegType.toStr('label')}")
        plt.ylabel(f"Rank")
        
        #plt.bar([rankList.index(rank) for rank in rankList], nonKRList, color='#4169E1', label = f'Non-KR {srcDegType.toStr("label")}')
        #plt.bar([rankList.index(rank) for rank in rankList], krList, color='#2E8B57', bottom=nonKRList, label = f'KR {srcDegType.toStr("label")}')

        plt.barh(xCoList, nonKRYCoList, color='#4169E1', label = f'Non-KR {srcDegType.toStr("label")}')
        plt.barh(xCoList, krYCoList, color='#2E8B57', left=nonKRYCoList, label = f'KR {srcDegType.toStr("label")}')
        
        plt.legend()

        figPath = waiter.WAITER.getPlotPath(f"NonKR_{srcDegType.toStr('label')}_{dstDegType.toStr('label')}_{argSizeOfCluster}", "Bar", self.field)

        plt.savefig(figPath)
        plt.clf()

    @error.callStackRoutine
    def plotLorentzCurve(self, argDegTuple, argIntegrate):

        if(not(isinstance(argDegTuple, tuple) and argIntegrate in [0, 1])):
            return None
        
        error.LOGGER.report("Plotting Lorentz Curve on # of Alumni", error.LogType.INFO)

        instDict = self.analyzer.getInstDict()
        numAlumniList = []

        srcDegType = argDegTuple[0]
        dstDegType = argDegTuple[1]

        for institution in util.getValuesListFromDict(instDict):
            if(institution.queryField(self.field)):
                numAlumni = 0

                for alumnus in institution.getTotalAlumniListForDeg(self.field, srcDegType):
                    if(alumnus.query(dstDegType)):
                        numAlumni+=1

                numAlumniList.append(numAlumni)

        if(1 == argIntegrate):

            error.LOGGER.report("Successfully Calculated Gini Coefficient on # of Alumni!", error.LogType.INFO)

            return util.calGiniCoeff(numAlumniList)

        (giniCoeff, xCoList, yCoList, baseList) = util.calGiniCoeff(numAlumniList)

        font = {'family': 'Helvetica', 'size': 9}

        plt.rc('font', **font)
        plt.figure(figsize=(7,5), dpi=200)

        titleStr = f"Lorentz Curve on Faculty Production (Field: {self.field})"
        ylabelStr = f"Cumulative Ratio Over Total {srcDegType.toStr('label')}_{dstDegType.toStr('label')} Production (Unit: Percentage)"

        plt.title(titleStr)
        plt.xlabel("Cumulative Ratio Over Total Number of Institutions (Unit: Percentage)")
        plt.ylabel(ylabelStr)

        plt.xlim(np.float32(0), np.float32(100))
        plt.ylim(np.float32(0), np.float32(100))

        plt.plot(xCoList, yCoList, 'bo-', markersize = 2)
        plt.plot(baseList, baseList, color = 'black', linewidth = 0.5)

        plt.fill_between(xCoList, yCoList, baseList, alpha = 0.2, color = 'grey')

        plt.text(60, 40, str(giniCoeff), color='black', fontsize=7)

        figPath = waiter.WAITER.getPlotPath(f"{srcDegType.toStr('label')}_{dstDegType.toStr('label')} Production", "LorentzCurve", self.field)
        plt.savefig(figPath)
        plt.clf()

        error.LOGGER.report("Successfully Calculated Gini Coefficient on # of Alumni!", error.LogType.INFO)

        return giniCoeff
    
    # ====================================================== #
    # 2.d. Methods for Extracting Network Data
    # ====================================================== #

    @error.callStackRoutine
    def __sortEdgeList(self):

        for edgeList in util.getValuesListFromDict(self.__edgeListDict):
            edgeList.sort_values(by=['# u', '# v'], inplace=True)

    @error.callStackRoutine
    def __sortVertexList(self):
        self.__vertexList.sort_values(by = ['# u'], inplace = True)

    @error.callStackRoutine
    def __exportXLSX(self):

        self.__sortVertexList()
        self.__sortEdgeList()

        vertexFilePath = waiter.WAITER.getCleanedFilePath("Vertex", self.field, util.FileExt.XLSX)
        self.__vertexList.to_excel(vertexFilePath, index = False, engine = 'openpyxl')

        for srcDstPair in util.getKeyListFromDict(self.__edgeListDict):

            srcLabel = srcDstPair[0].toStr("label")
            dstLabel = srcDstPair[1].toStr("label")

            edgeFilePath = waiter.WAITER.getCleanedFilePath("Edge", self.field, util.FileExt.XLSX, srcLabel, dstLabel)

            self.__edgeListDict[srcDstPair].to_excel(edgeFilePath, index= False, engine = 'openpyxl')
    

    @error.callStackRoutine
    def __exportCSV(self):

        self.__sortVertexList()
        self.__sortEdgeList()

        vertexFilePath = waiter.WAITER.getCleanedFilePath("Vertex", self.field, util.FileExt.CSV)
        self.__vertexList.to_csv(vertexFilePath, index = False, sep = '\t')

        for srcDstPair in util.getKeyListFromDict(self.__edgeListDict):

            srcLabel = srcDstPair[0].toStr("label")
            dstLabel = srcDstPair[1].toStr("label")

            edgeFilePath = waiter.WAITER.getCleanedFilePath("Edge", self.field, util.FileExt.CSV, srcLabel, dstLabel)

            self.__edgeListDict[srcDstPair].to_csv(edgeFilePath, index= False, sep = '\t')

    @error.callStackRoutine
    def exportVertexAndEdgeList(self, argFileExtension: util.FileExt):

        if(util.FileExt.XLSX == argFileExtension):
            self.__exportXLSX()
        elif(util.FileExt.CSV == argFileExtension):
            self.__exportCSV()
        else:
            error.LOGGER.report(": ".join(["Invalid File Extension", util.fileExtToStr(argFileExtension)], error.LogType.ERROR))


    @error.callStackRoutine
    def exportRankList(self, argTargetDeg: career.DegreeType, argFileExtension: util.FileExt):

        if(not isinstance(argTargetDeg, career.DegreeType)):
            error.LOGGER.report(f"Invalid argTargetDeg", error.LogType.ERROR)
            return None
        elif(not isinstance(argFileExtension, util.FileExt)):
            error.LOGGER.report(f"Invalid argFileExtension", error.LogType.ERROR)
            return None


        instList = util.getValuesListFromDict(self.analyzer.getInstDict())

        rank2InstDict = {}

        for inst in instList:
            department = inst.getFieldIfExists(self.field)

            if(None != department):    #institution has self.field field

                rank = inst.getRankAt(self.field, argTargetDeg)

                if(None != rank):   #rank is valid
                    instInfoStrList = []
                    inst.flushInfoToList(instInfoStrList)

                    rank2InstDict[rank] = ', '.join(instInfoStrList[1:])

        sortedRankList = sorted(util.getKeyListFromDict(rank2InstDict))
        sortedInstInfoStrList = [rank2InstDict[rank] for rank in sortedRankList]

        dataDict = {"Rank": sortedRankList,
                    "Inst": sortedInstInfoStrList}

        rankDf = pd.DataFrame(dataDict)
        filePath = waiter.WAITER.getCleanedFilePath("RankList", self.field, argFileExtension, argTargetDeg.toStr("label"))

        if(util.FileExt.XLSX == argFileExtension):
            return rankDf.to_excel(filePath, index= False, engine = 'openpyxl')
        elif(util.FileExt.CSV == argFileExtension):
            return rankDf.to_csv(filePath, sep='\t', index=False)

if(__name__ == '__main__'):

    error.LOGGER.report("This Module is Not for Main Function", error.LogType.CRITICAL)

    
