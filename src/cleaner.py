"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""

import pandas as pd
import datetime as dt
import os
import error
import util
import institution
import numpy as np
import SpringRank as sp

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
            #print(argTargetRow[index])
            if(util.areSameStrings(argTargetRow[index], 'PhD')):   

                instInfo = institution.InstInfo()
                if(not instInfo.addRegionToInstInfo(argTargetRow[index+2])): #better version
                    continue

                if(instInfo.isInvalid()):
                    continue

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
            argRowIterator.changeTargetAndReset('Degree')
            error.LOGGER.report("This Row does not Contain PhD or AP Info", error.LogType.WARNING)
            return 0
        
        currentRank = argTargetRow[8]
        gender = argTargetRow[9]
        
        edgeRow.append(phDInst.instId)
        edgeRow.append(apInst.instId)
        edgeRow.append(gender)

        #add alumnus info
        phDInst.addAlumnusAt(self.field, apInst.instId, currentRank, gender)


        if(1 == newVertexRowsCreated):
                for vertexRow in vertexRowList:
                    self.__vertexList.loc[len(self.__vertexList)] = vertexRow
            
        self.__edgeList.loc[len(self.__edgeList)] = edgeRow

        argRowIterator.changeTargetAndReset('Degree')

        error.LOGGER.report("Successfully Cleaned a Row!", error.LogType.INFO)

        return 1

    @error.callStackRoutine
    def __sortEdgeList(self):
        self.__edgeList.sort_values(by=['# u', '# v'], inplace=True)

    @error.callStackRoutine
    def __sortVertexList(self):
        self.__vertexList.sort_values(by = ['# u'], inplace = True)

    @error.callStackRoutine
    def __genDestDir(self):
        if(False == os.path.isdir('../dataset/cleaned')):
            os.mkdir('../dataset/cleaned')
            error.LOGGER.report("No Directory named 'dataset_cleaned', thus a New One is Generated", error.LogType.INFO)

    @error.callStackRoutine
    def calcSpringRank(self):

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

    
        spRankList = util.getRankBasedOn(sp.SpringRank(adjMat, 2))

        for index in range(len(spRankList)):
            instId = index2IdMap[index]

            self.analyzer.getExistingInstitution(instId).setMVRRankAt(self.field, spRankList[index])

        error.LOGGER.report("Sucessfully Calculated SpringRank!", error.LogType.INFO)

        return 1

    @error.callStackRoutine
    def calcGiniCoeff(self):
        
        error.LOGGER.report("Calculating Gini Coefficient on # of Alumni", error.LogType.INFO)
        error.LOGGER.report("Result of This Method Can be Unreliable", error.LogType.WARNING)

        instDict = self.analyzer.getInstDict()
        numAlumniList = []

        for institution in util.getValuesListFromDict(instDict):
            if(institution.queryField(self.field)):
                numAlumniList.append(institution.getTotalNumAlumniAt(self.field))

        returnValue = util.calGiniCoeff(numAlumniList)
        error.LOGGER.report("Successfully Calculated Gini Coefficient on # of Alumni!", error.LogType.INFO)

        return returnValue

    @error.callStackRoutine
    def calcMVRRank(self):
        
        error.LOGGER.report("Calculating MVR Rank", error.LogType.INFO)

        returnValue = self.calcSpringRank()

        error.LOGGER.report("Sucessfully Calculated MVR Rank!", error.LogType.INFO)
        return returnValue
    
    @error.callStackRoutine
    def __exportXLSX(self):

        currentDateStr = dt.datetime.now().strftime("%Y_%m_%d")

        self.__sortVertexList()
        self.__sortEdgeList()

        vertexFilename = ''.join(['_'.join(['vertex', self.field, currentDateStr]), '.xlsx'])
        edgeFilename = ''.join(['_'.join(['edge', self.field, currentDateStr]), '.xlsx'])

        vertexFilePath = os.path.join('../dataset/cleaned/', vertexFilename)
        edgeFilePath = os.path.join('../dataset/cleaned/', edgeFilename)
        

        self.__vertexList.to_excel(vertexFilePath, index = False, engine = 'openpyxl')
        self.__edgeList.to_excel(edgeFilePath, index= False, engine = 'openpyxl')
    

    @error.callStackRoutine
    def __exportCSV(self):

        currentDateStr = dt.datetime.now().strftime("%Y_%m_%d")

        self.__sortVertexList()
        self.__sortEdgeList()

        vertexFilename = ''.join(['_'.join(['vertex', self.field, currentDateStr]), '.csv'])
        edgeFilename = ''.join(['_'.join(['edge', self.field, currentDateStr]), '.csv'])

        vertexFilePath = os.path.join('../dataset/cleaned/', vertexFilename)
        edgeFilePath = os.path.join('../dataset/cleaned/', edgeFilename)

        self.__vertexList.to_csv(vertexFilePath, index = False, sep = '\t')
        self.__edgeList.to_csv(edgeFilePath, index = False, sep = '\t')

    @error.callStackRoutine
    def exportVertexAndEdgeListAs(self, argFileExtension: util.FileExt):

        self.__genDestDir()

        if(util.FileExt.XLSX == argFileExtension):
            self.__exportXLSX()
        elif(util.FileExt.CSV == argFileExtension):
            self.__exportCSV()
        else:
            error.LOGGER.report(": ".join(["Invalid File Extension", util.fileExtToStr(argFileExtension)], error.LogType.ERROR))

if(__name__ == '__main__'):

    error.LOGGER.report("This Module is Not for Main Function", error.LogType.CRITICAL)

    
