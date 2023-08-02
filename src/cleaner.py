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
import career

import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import quad

class CleanerFlag():

    @error.callStackRoutine
    def __init__(self):
        self.MVRRankCalculated = False
        self.giniCoeffCalculated = False

    @error.callStackRoutine
    def raiseMVRRankCalculated(self):
        self.MVRRankCalculated = True

    @error.callStackRoutine
    def ifMVRRankCalculated(self):
        return self.MVRRankCalculated
    
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
    def __init__(self, argAnalyzer, argField, argKorea):

        self.analyzer= argAnalyzer

        # Field Name
        self.field = argField

        # Dataframe of output vertex list
        self.__vertexList = pd.DataFrame(columns = ['# u', 'Institution', 'Region', 'Country'])

        self.__edgeListDict = {}

        for srcDegType in career.DegreeType:

            dstDegType = srcDegType.next()

            if(None != dstDegType):
                self.__edgeListDict[(srcDegType, dstDegType)] = pd.DataFrame(columns = ['# u', '# v', 'gender'])

        self.__edgeListDict[(career.DegreeType.PHD, career.DegreeType.AP)] = pd.DataFrame(columns = ['# u', '# v', 'gender'])

        # Dictonary holding institution ids, which are matched to institutions one by one.
        self.__localInstIdList = []

        self.korea = argKorea

        # Flags.
        self.flags = CleanerFlag()

    @error.callStackRoutine
    def ifKoreaOnly(self):
        return self.korea
    
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
    def __cleanStep(self, argTargetRow, argRowIterator: util.rowIterator, argDegType: career.DegreeType):

        hit = 0
        step = None
        newVertexRowCreated = 0
        vertexRow = []

        offset = 2

        key = argDegType.toStr("search")

        koreaOnly = self.ifKoreaOnly()

        if(argDegType in [career.DegreeType.PDOC, career.DegreeType.AP]):
            argRowIterator.changeTargetAndReset('Job')
            offset = 1

        for index in argRowIterator:

            if(util.areSameStrings(argTargetRow[index], key)):   

                instInfo = institution.InstInfo()

                if(not instInfo.addRegionToInstInfo(argTargetRow[index+offset])): #better version
                    continue

                if(instInfo.isInvalid(koreaOnly)):
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

        targetDegList = [career.DegreeType.BS, career.DegreeType.PHD, career.DegreeType.AP] if (self.ifKoreaOnly()) else career.DegreeType

        # Step 1: Inspect the target row and gather informations.

        for degType in targetDegList:
            #Post-doc case is not complete (will not work maybe..)

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

        alumnus = career.Alumni(self.field, currentRank, util.strToGender(gender))

        for step in util.getValuesListFromDict(careerStepDict):
            alumnus.append(step)

        # Step 3: Maintain vertex list and edge list.

        for vertexRow in vertexRowList:
            self.__vertexList.loc[len(self.__vertexList)] = vertexRow

        if(not self.ifKoreaOnly()):
            for srcDegType in targetDegList:

                dstDegType = srcDegType.next()

                if(None != dstDegType):
                    if(hitDict[srcDegType] and hitDict[dstDegType]):
                        
                        targetEdgeList = self.__edgeListDict[(srcDegType, dstDegType)]
                        edgeRow = []

                        srcInstId = careerStepDict[srcDegType].inst.instId
                        dstInstId = careerStepDict[dstDegType].inst.instId

                        edgeRow.append(srcInstId)
                        edgeRow.append(dstInstId)
                        edgeRow.append(gender)

                        targetEdgeList.loc[len(targetEdgeList)] = edgeRow

                    else:
                        continue


        if(hitDict[career.DegreeType.PHD] and hitDict[career.DegreeType.AP]):
                    
            targetEdgeList = self.__edgeListDict[(career.DegreeType.PHD, career.DegreeType.AP)]
            edgeRow = []

            srcInstId = careerStepDict[career.DegreeType.PHD].inst.instId
            dstInstId = careerStepDict[career.DegreeType.AP].inst.instId

            edgeRow.append(srcInstId)
            edgeRow.append(dstInstId)
            edgeRow.append(gender)

            targetEdgeList.loc[len(targetEdgeList)] = edgeRow

        argRowIterator.changeTargetAndReset('Degree')

        error.LOGGER.report("Successfully Cleaned a Row!", error.LogType.INFO)

        status.STATTRACKER.statTotalNumRowCleaned += 1
        return 1

    @error.callStackRoutine
    def __sortEdgeList(self):

        for edgeList in util.getValuesListFromDict(self.__edgeListDict):
            edgeList.sort_values(by=['# u', '# v'], inplace=True)

    @error.callStackRoutine
    def __sortVertexList(self):
        self.__vertexList.sort_values(by = ['# u'], inplace = True)

    @error.callStackRoutine
    def __calcSpRankForStep(self, argDegTuple):

        srcDegType = argDegTuple[0]
        dstDegType = argDegTuple[1]

        if(srcDegType not in career.DegreeType or srcDegType not in career.DegreeType):
            error.LOGGER.report("Invalid argDegTyple.", error.LogType.WARNING)
            return None

        targetEdgeList = self.__edgeListDict[(srcDegType, dstDegType)]

        adjMat = np.zeros((len(self.__localInstIdList), len(self.__localInstIdList)), dtype = int)

        id2IndexMap = {}
        index2IdMap = {}

        targetRow = None

        localInstIdListSorted = sorted(self.__localInstIdList)

        for i in range(len(localInstIdListSorted)):
            id2IndexMap[localInstIdListSorted[i]] = i
            index2IdMap[i] = localInstIdListSorted[i]

        for numRow in range(len(targetEdgeList.index)):
            targetRow = targetEdgeList.iloc[numRow]
            adjMat[id2IndexMap[targetRow[0]], id2IndexMap[targetRow[1]]] += 1

        spRankList = sp.get_ranks(adjMat)
        inverseTemp = sp.get_inverse_temperature(adjMat, spRankList)

        rankList = util.getRankBasedOn(spRankList)
        rankList = util.getRidOfTie(rankList)

        for index in range(len(rankList)):
            instId = index2IdMap[index]

            self.analyzer.getExistingInstitution(instId).setRankAt(self.field, rankList[index], util.RankType.SPRANK, argDegTuple)

        return inverseTemp

    @error.callStackRoutine
    def calcRank(self):

        returnDict = {}
        
        for rankType in self.analyzer.getRankTypeList():
            if(util.RankType.SPRANK == rankType):

                error.LOGGER.report("Calculating SpringRank, which is an approximation of MVR Rank", error.LogType.INFO)
                error.LOGGER.report("Copyright (c) 2017 Caterina De Bacco and Daniel B Larremore", error.LogType.INFO)

                returnDict[rankType] = []

                for srcDegType in career.DegreeType:

                    dstDegType = srcDegType.next()

                    if(None != dstDegType):
                        returnDict[rankType].append(self.__calcSpRankForStep((srcDegType, dstDegType)))

                returnDict[rankType].append(self.__calcSpRankForStep((career.DegreeType.PHD, career.DegreeType.AP)))

                error.LOGGER.report("Sucessfully Calculated SpringRank!", error.LogType.INFO)
            
            elif(util.RankType.JRANK == rankType):

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
                        inst.setRankAt(self.field, rank, util.RankType.JRANK, None)

                returnDict[rankType] = 1

        self.flags.raiseMVRRankCalculated()

        return returnDict

    @error.callStackRoutine
    def calcGiniCoeff(self):

        print("GiniCoeff")
        
        error.LOGGER.report("Calculating Gini Coefficient on # of Alumni", error.LogType.INFO)
        error.LOGGER.report("Result of This Method Can be Unreliable", error.LogType.WARNING)

        instDict = self.analyzer.getInstDict()
        numAlumniList = []

        for institution in util.getValuesListFromDict(instDict):
            if(institution.queryField(self.field)):
                numAlumniList.append(institution.getTotalNumAlumniAt(self.field, career.DegreeType.PHD))

        returnValue = util.calGiniCoeff(numAlumniList, "Faculty Production", self.field)

        #print(self.field)
        #print(numAlumniList)

        self.flags.raiseGiniCoeffCalculated()

        error.LOGGER.report("Successfully Calculated Gini Coefficient on # of Alumni!", error.LogType.INFO)

        return returnValue
    
    @error.callStackRoutine
    def calcGiniCoeffBS(self):

        instDict = self.analyzer.getInstDict()
        
        numAlumniDict = {}

        for inst in util.getValuesListFromDict(instDict):
            department = inst.getFieldIfExists(self.field)

            if(None != department and not inst.isNonKRInst()):
                numAlumniDict[inst.instId] = 0

        for dstInst in util.getValuesListFromDict(instDict):
            
            dstDepartment = dstInst.getFieldIfExists(self.field)

            if(None != dstDepartment):
                for alumnus in dstDepartment.getTotalAlumniListForDeg(career.DegreeType.AP):
                        
                        srcInst = alumnus.getInstFor(career.DegreeType.PHD)

                        if(None == srcInst or srcInst.isNonKRInst()):
                            continue
                        else:
                            numAlumniDict[srcInst.instId] += 1

        numAlumniList = util.getValuesListFromDict(numAlumniDict)

        returnValue = util.calGiniCoeff(numAlumniList, "Faculty Production", self.field)

        self.flags.raiseGiniCoeffCalculated()

        return returnValue

    
    @error.callStackRoutine
    def calcAvgMVRMoveBasedOnGender(self, argGender: util.Gender):

        if(not self.flags.ifMVRRankCalculated()):
            error.LOGGER.report("Attempt denied. MVR ranks are not pre-calculated.", error.LogType.ERROR)
            return 0
        
        if(argGender not in util.Gender):
            error.LOGGER.report("Invalid Gender", error.LogType.WARNING)
            return 0

        instDict = self.analyzer.getInstDict()
        rankMovementList = []

        for inst in util.getValuesListFromDict(instDict):
            
            department = inst.getFieldIfExists(self.field)

            if(None != department):
                for alumnus in department.getTotalAlumniListForDeg(career.DegreeType.PHD):
                    if(argGender == alumnus.gender):
                        rankMovementList.append(alumnus.getRankMove(util.RankType.SPRANK, (career.DegreeType.PHD, career.DegreeType.AP)))


        return util.getMean(rankMovementList)
    
    @error.callStackRoutine
    def calcAvgMVRMoveForRange(self, argPercentLow: int, argPercentHigh: int):

        if(not self.flags.ifMVRRankCalculated()):
            error.LOGGER.report("Attempt denied. MVR ranks are not pre-calculated.", error.LogType.ERROR)
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
                rankList.append(inst.getRankAt(self.field, util.RankType.SPRANK, (career.DegreeType.PHD, career.DegreeType.AP)))

        rankLowerBound = math.ceil(float(len(rankList) * argPercentLow / 100))
        rankUpperBound = math.floor(float(len(rankList)* argPercentHigh / 100))

        
        instListSorted = util.sortObjectBasedOn(instList, rankList)
        
        for inst in instListSorted[rankLowerBound:rankUpperBound+1]:
            department = inst.getFieldIfExists(self.field)

            if(None != department):    #institution has self.field field
                for alumni in department.getTotalAlumniListForDeg(career.DegreeType.PHD):
                    for alumnus in alumni:

                        rankMovementList.append(alumnus.getRankMove(util.RankType.SPRANK, (career.DegreeType.PHD, career.DegreeType.AP)))

        return util.getMean(rankMovementList)

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

                for alumnus in department.getTotalAlumniListForDeg(career.DegreeType.PHD):

                    rankMove = alumnus.getRankMove(util.RankType.SPRANK, (career.DegreeType.PHD, career.DegreeType.AP))

                    if(None == rankMove):
                        continue

                    data = alumnus.getRankMove(util.RankType.SPRANK, (career.DegreeType.PHD, career.DegreeType.AP)) / numTotalInst

                    phDInstRank = alumnus.getInstFor(career.DegreeType.PHD).getRankAt(self.field, util.RankType.SPRANK, (career.DegreeType.PHD, career.DegreeType.AP))

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

        figPath = util.getPlotPath("GenderRatio", "Plot", self.field)

        plt.savefig(figPath)
        plt.clf()

        return 1
    
    @error.callStackRoutine
    def plotNonKRFac(self, argDegType: career.DegreeType):

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

        for dstInst in util.getValuesListFromDict(instDict):
            
            dstDepartment = dstInst.getFieldIfExists(self.field)

            if(None != dstDepartment):
                for rankType in rankTypeList:
                    
                    dstInstRank = dstInst.getRankAt(self.field, rankType, (career.DegreeType.PHD, career.DegreeType.AP))

                    if(None == dstInstRank):
                        continue

                    if(dstInstRank not in nonKRProfDict[rankType]):
                        nonKRProfDict[rankType][dstInstRank] = 0

                    for alumnus in dstDepartment.getTotalAlumniListForDeg(career.DegreeType.AP):
                        
                        srcInst = alumnus.getInstFor(argDegType)

                        if(None == srcInst):
                            continue
                        
                        isNonKR = srcInst.isNonKRInst()

                        nonKRProfDict[rankType][dstInstRank] += isNonKR

        font = {'family': 'serif', 'size': 9}
        titleStr = "Number of Faculty with Non-KR "+ argDegType.toStr("label") + " (Field: " + self.field + ")"

        for rankType in rankTypeList:

            for rank in sorted(util.getKeyListFromDict(nonKRProfDict[rankType])):

                xCoDict[rankType].append(rank)
                yCoDict[rankType].append(nonKRProfDict[rankType][rank])
            
            plt.rc('font', **font)

            plt.title(titleStr)
            plt.xlabel(rankType.toStr("camelcase"))
            plt.ylabel(f"Number of Faculty with Non-KR {argDegType.toStr('label')}")

            plt.plot(xCoDict[rankType], yCoDict[rankType], color='blue')

            plt.legend()

            figPath = util.getPlotPath(f"NumNonKRFac_{argDegType.toStr('label')}", "Plot", self.field, rankType.toStr("abbrev"))

            plt.savefig(figPath)
            plt.clf()

        return 1
    
    @error.callStackRoutine
    def plotNonKRFac2(self, argDegType: career.DegreeType):

        if(not self.flags.ifMVRRankCalculated()):
            error.LOGGER.report("Attempt denied. MVR ranks are not pre-calculated.", error.LogType.ERROR)

            return 0

        instDict = self.analyzer.getInstDict()
        rankTypeList = self.analyzer.getRankTypeList()

        print(rankTypeList)

        nonKRProfDict = {}
        krProfDict = {}
        xCoDict = {}
        yCoDict = {}
        yCoDict2 = {}

        for rankType in rankTypeList:
            nonKRProfDict[rankType] = {}
            krProfDict[rankType] = {}
            xCoDict[rankType] = []
            yCoDict[rankType] = []
            yCoDict2[rankType] = []

        for dstInst in util.getValuesListFromDict(instDict):
            
            dstDepartment = dstInst.getFieldIfExists(self.field)

            if(None != dstDepartment):
                for rankType in rankTypeList:
                    
                    dstInstRank = dstInst.getRankAt(self.field, rankType, (career.DegreeType.PHD, career.DegreeType.AP))

                    if(None == dstInstRank):
                        continue
                    elif(not dstInst.isNonKRInst()):
                        continue
                    
                    #print(rankType, ": nonKR :" , nonKRProfDict[rankType])
                    if(dstInstRank not in nonKRProfDict[rankType]):
                        #print(f"{dstInstRank} is not in")
                        nonKRProfDict[rankType][dstInstRank] = 0

                    #print(rankType, ": KR : " ,  krProfDict[rankType])
                    if(dstInstRank not in krProfDict[rankType]):
                        #print(f"{dstInstRank} is not in")
                        krProfDict[rankType][dstInstRank] = 0

                    for alumnus in dstDepartment.getTotalAlumniListForDeg(career.DegreeType.AP):
                        
                        srcInst = alumnus.getInstFor(argDegType)

                        if(None == srcInst):
                            continue
                        
                        isNonKR = srcInst.isNonKRInst()
                        
                        if(1 == isNonKR):
                            nonKRProfDict[rankType][dstInstRank] += 1
                        elif(0 == isNonKR):
                            krProfDict[rankType][dstInstRank] += 1

                        print(isNonKR, srcInst.getRankAt(self.field, rankType, (career.DegreeType.PHD, career.DegreeType.AP)),\
                              dstInstRank)

                        #print(nonKRProfDict)
                        #print(krProfDict)

        font = {'family': 'serif', 'size': 9}
        titleStr = "Number of Faculty with Non-KR "+ argDegType.toStr("label") + " (Field: " + self.field + ")"

        print(nonKRProfDict[util.RankType.SPRANK])
        print(krProfDict[util.RankType.SPRANK])
        for rankType in rankTypeList:

            for rank in sorted(util.getKeyListFromDict(nonKRProfDict[rankType])):

                xCoDict[rankType].append(rank)
                yCoDict[rankType].append(nonKRProfDict[rankType][rank])
                yCoDict2[rankType].append(krProfDict[rankType][rank])
            
            plt.rc('font', **font)

            plt.title(titleStr)
            plt.xlabel(rankType.toStr("camelcase"))
            plt.ylabel(f"Number of Faculty with Non-KR {argDegType.toStr('label')}")

            plt.plot(xCoDict[rankType], yCoDict[rankType], color='blue')
            plt.plot(xCoDict[rankType], yCoDict2[rankType], color='green')

            plt.legend()

            figPath = util.getPlotPath(f"NumNonKRFac_{argDegType.toStr('label')}", "Plot", self.field, rankType.toStr("abbrev"))

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

                    phDInstRank = inst.getRankAt(self.field, rankType, (career.DegreeType.PHD, career.DegreeType.AP))

                    if(None != phDInstRank and phDInstRank not in nonKRProfDict[rankType]):
                        nonKRProfDict[rankType][phDInstRank] = 0

                    if(None != phDInstRank and phDInstRank not in profDict[rankType]):
                        profDict[rankType][phDInstRank] = 0
                    
                    isNonKRPhD = inst.isNonKRInst()

                    for alumnus in department.getTotalAlumniListForDeg(career.DegreeType.PHD):

                        apInstRank = alumnus.getInstFor(career.DegreeType.AP).getRankAt(self.field, rankType, (career.DegreeType.PHD, career.DegreeType.AP))

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

        vertexFilePath = util.getCleanedFilePath("Vertex", self.field, util.FileExt.XLSX)
        self.__vertexList.to_excel(vertexFilePath, index = False, engine = 'openpyxl')

        for srcDstPair in util.getKeyListFromDict(self.__edgeListDict):

            srcLabel = srcDstPair[0].toStr("label")
            dstLabel = srcDstPair[1].toStr("label")

            edgeFilePath = util.getCleanedFilePath("Edge", self.field, util.FileExt.XLSX, srcLabel, dstLabel)

            self.__edgeListDict[srcDstPair].to_excel(edgeFilePath, index= False, engine = 'openpyxl')
    

    @error.callStackRoutine
    def __exportCSV(self):

        self.__sortVertexList()
        self.__sortEdgeList()

        vertexFilePath = util.getCleanedFilePath("Vertex", self.field, util.FileExt.CSV)
        self.__vertexList.to_csv(vertexFilePath, index = False, sep = '\t')

        for srcDstPair in util.getKeyListFromDict(self.__edgeListDict):

            srcLabel = srcDstPair[0].toStr("label")
            dstLabel = srcDstPair[1].toStr("label")

            edgeFilePath = util.getCleanedFilePath("Edge", self.field, util.FileExt.CSV, srcLabel, dstLabel)

            self.__edgeListDict[srcDstPair].to_csv(edgeFilePath, index= False, sep = '\t')

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

    
