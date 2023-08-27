"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""

import util
import error
from enum import Enum, auto
import status
import career
import setting

class InstInfo:

    @error.callStackRoutine
    def __init__(self):
        self.instId: int = None
        self.name: str = None
        self.region: str = None
        self.country: str = None    

    #better way for institution location splitting (strongly recommended)
    @error.callStackRoutine
    def addRegionToInstInfo(self, argInstLocationInfo: str) -> None:

        if(str != type(argInstLocationInfo)):
            return 0
        instLocation:list = argInstLocationInfo.split(',')
        
        if(3 != len(instLocation)):
            error.LOGGER.report("Invalid Institution Info Detected.", error.LogType.WARNING)
            return 0
    

        self.name = instLocation[0].strip().lower()
        self.region = instLocation[1].strip().lower()
        self.country = instLocation[2].strip().upper() 

        return 1

    @error.callStackRoutine
    def addInstIdToInstInfo(self, argInstId: int) -> None:
        self.instId = argInstId

    @error.callStackRoutine
    def returnKeyTuple(self) -> tuple:
        
        return (self.name, self.region, self.country)

    @error.callStackRoutine
    def isInvalid(self):

        if(None!= self.instId and util.isEmptyData(self.instId)):
            return 1
        elif(util.isEmptyData(self.name)):
            return 1
        elif(util.isEmptyData(self.region)):
            return 1
        elif(util.isEmptyData(self.country)):
            return 1
        elif("CLOSED" == setting.PARAM["Basic"]["networkType"]):
            if(util.areSameStrings(self.country, 'KR')):
                if(self.region in ['blacksburg', 'ann arbor', 'austin']):

                    return 1
                else:
                    return 0
            else:
                return 1    
        else:
            return 0
    
    @error.callStackRoutine
    def print(self) -> None: #for debugging purpose

        print("==========================")
        print(self.instId)
        print(self.name)
        print(self.region)
        print(self.country)
        print("==========================")

class Institution:

    @error.callStackRoutine
    def __init__(self, argInstInfo: InstInfo):

        self.instId = argInstInfo.instId
        self.name = argInstInfo.name
        self.region = argInstInfo.region
        self.country = argInstInfo.country
        self.fieldDict = {}

        status.STATTRACKER.statTotalNumInst+=1

        del argInstInfo

    @error.callStackRoutine
    def printInfo(self, argGran):

        print("===========================================================")

        print("Institution Id: ", self.instId)
        print("Name: ", self.name)
        print("Region: ", self.region)
        print("Country: ", self.country)
        print("")

        if("inst" == argGran):
            return
        
        for item in sorted(self.fieldDict.items()):

            department = self.fieldDict[item[0]]
            print(department)

            if("field" == argGran):
                continue
            
            department.printAlumniInfo()


        print("===========================================================")

    @error.callStackRoutine
    def __repr__(self, argGran) -> str:
        strList = []

        strList.append("===========================================================")

        strList.append(f"Institution Id: {self.instId}")
        strList.append(f"Name: {self.name}")
        strList.append(f"Region: {self.region}")
        strList.append(f"Country: {self.country}")
        strList.append("")

        if("inst" == argGran):
            return '\n'.join(strList)
        
        for item in sorted(self.fieldDict.items()):

            department = self.fieldDict[item[0]]
            strList.append(department.__repr__())

            if("field" == argGran):
                continue
            
            department.printAlumniInfo()
        
        return '\n'.join(strList)

    @error.callStackRoutine
    def flushInfoToList(self, argList: list):
        argList.append(self.instId)
        argList.append(self.name)
        argList.append(self.region)
        argList.append(self.country)

    @error.callStackRoutine
    def getInfoString(self):
        return f"{self.name}, {self.region}, {self.country}"

    @error.callStackRoutine
    def getField(self, argField):

        if(argField not in self.fieldDict):
            self.fieldDict[argField] = InstituionAtField(argField)
            
            return self.fieldDict[argField]
        
        return self.fieldDict[argField]
    
    @error.callStackRoutine
    def queryField(self, argField):
        return int(argField in self.fieldDict)
    
    @error.callStackRoutine
    def getFieldIfExists(self, argField):
        if(self.queryField(argField)):
            return self.getField(argField)
        else:
            return None
    
    @error.callStackRoutine
    def setRankAt(self, argField, argRank, argDegType: career.DegreeType):

        if(argDegType not in setting.PARAM["Basic"]["targetDeg"]):
            error.LOGGER.report("Invalid argDegType", error.LogType.ERROR)

        self.getField(argField).rankDict[argDegType] = argRank

    @error.callStackRoutine
    def getRankAt(self, argField, argDegType: career.DegreeType):

        if(argDegType not in setting.PARAM["Basic"]["targetDeg"]):
            error.LOGGER.report("Invalid argDegType", error.LogType.ERROR)

        return self.getField(argField).rankDict[argDegType]

    @error.callStackRoutine
    def addAlumnus(self, argAlumnus: career.Alumni, argDegType: career.DegreeType):

        if(self.queryField(argAlumnus.field)):
            return self.getField(argAlumnus.field).addAlumnus(argAlumnus, argDegType)
        
        return None

    @error.callStackRoutine
    def getAlumniAt(self, argField, argDestInstitutionId, argDegType):

        if(self.queryField(argField)):
            return self.getField(argField).getAlumniWentTo(argDestInstitutionId, argDegType)
        
        return None

    @error.callStackRoutine
    def getNumAlumniAt(self, argField, argDestInstitutionId, argDegType):

        if(self.queryField(argField)):
            return self.getField(argField).getNumAlumniWentTo(argDestInstitutionId, argDegType)
        
        return 0

    @error.callStackRoutine    
    def getTotalNumAlumniAt(self, argField, argDegType):

        if(self.queryField(argField)):
            return self.getField(argField).getTotalNumAlumniForDeg(argDegType)
                
        return 0
    
    @error.callStackRoutine
    def getTotalAlumniListForDeg(self, argField, argDegType):
        if(self.queryField(argField)):
            return self.getField(argField).getTotalAlumniListForDeg(argDegType)
    
    @error.callStackRoutine
    def getTotalNumAlumniForDeg(self, argDegType: career.DegreeType):

        totalNumAlumni = 0

        for field in list(self.fieldDict.values()):

            totalNumAlumni += field.getTotalNumAlumniForDeg(argDegType)

        return totalNumAlumni
    
    @error.callStackRoutine
    def isNonKRInst(self):
        return int(not util.areSameStrings(self.country, 'KR'))


class InstituionAtField(Institution):

    @error.callStackRoutine
    def __init__(self, argField):

        self.field = argField

        self.rankDict = {}
        self.alumniDict = {}

        for degType in setting.PARAM["Basic"]["targetDeg"]:
            self.alumniDict[degType] = []
            self.rankDict[degType] = None

        status.STATTRACKER.statFieldNumInst[self.field] += 1

    @error.callStackRoutine
    def __repr__(self) -> str:
        strList = []

        strList.append("")
        strList.append(f"=== Field: {self.field}===")
        strList.append(f"Rank: {self.rankDict}")
        strList.append("")
        
        return '\n'.join(strList)
    
    @error.callStackRoutine
    def printAlumniInfo(self):

        for degType in setting.PARAM["Basic"]["targetDeg"]:

            targetAlumniList = self.alumniDict[degType]

            print(f">>>   {degType.toStr('label')}   <<<")
            print("Number Of Alumni: ", self.getTotalNumAlumniForDeg(degType))
        
            for alumnus in targetAlumniList:
                print(alumnus)

    @error.callStackRoutine
    def addAlumnus(self, argAlumnus: career.Alumni, argDegType: career.DegreeType):

        if(argDegType not in setting.PARAM["Basic"]["targetDeg"]):
            return 0
        elif(not isinstance(argAlumnus, career.Alumni)):
            return 0
        
        self.alumniDict[argDegType].append(argAlumnus)

        return 1

    @error.callStackRoutine
    def getAlumniWentTo(self, argDestInstitutionId, argDegType: career.DegreeType):

        alumniList = []

        for alumnus in self.alumniDict[argDegType]:

            step = alumnus.query(argDegType)

            if(None != step):
                if(argDestInstitutionId == step.inst.instId):
                    alumniList.append(alumnus)

        return alumniList
    
    @error.callStackRoutine
    def getNumAlumniWentTo(self, argDestInstitutionId, argDegType: career.DegreeType):
        return len(self.getAlumniWentTo(argDestInstitutionId, argDegType))

    @error.callStackRoutine
    def getTotalNumAlumniForDeg(self, argDegType: career.DegreeType):
        return len(self.getTotalAlumniListForDeg(argDegType))
    
    @error.callStackRoutine
    def getTotalAlumniListForDeg(self, argDegType: career.DegreeType):
        return self.alumniDict[argDegType]


if(__name__ == "__main__"):
    
    error.LOGGER.report("This Module is Not for Main Function", error.LogType.CRITICAL)

