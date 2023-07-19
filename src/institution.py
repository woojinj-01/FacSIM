"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""

import util
import error
from enum import Enum, auto
import status

class RankType(Enum):

    SPRANK = auto()
    JRANK = auto()

    @error.callStackRoutine
    def toStr(self, argRepType):
        if("camelcase"):
            match self:
                case RankType.SPRANK:
                    return "SpringRank"
                case RankType.JRANK:
                    return "Joongang-ilbo Rank"
                case _:
                    pass

        elif("abbrev"):
            return self.name.lower()
        
        else:
            error.LOGGER.report("Wrong argRepType.", error.LogType.ERROR)


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


class AlumnusInfo:

    @error.callStackRoutine
    def __init__(self, argField: str, argCurrentRank: str, argGender: util.Gender, argPhDInstId, argPhDInstName, argApInstId, argApInstName):
        self.field: str = argField
        self.currentRank: str = argCurrentRank
        self.gender: util.Gender = argGender
        self.phDInstId: int = argPhDInstId
        self.phDInstName: str = argPhDInstName
        self.apInstId: int = argApInstId
        self.apInstName: str = argApInstName
        
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
            department.printInfo()

            if("field" == argGran):
                continue
            
            department.printAlumniInfo()


        print("===========================================================")

    @error.callStackRoutine
    def flushInfoToList(self, argList: list):
        argList.append(self.instId)
        argList.append(self.name)
        argList.append(self.region)
        argList.append(self.country)

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
    def setRankAt(self, argField, argRank, argRankType):
        if(argRankType not in [RankType.SPRANK, RankType.JRANK]):
            error.LOGGER.report("Invalid argRankType", error.LogType.ERROR)
        
        if(RankType.SPRANK == argRankType):
            self.getField(argField).spRank = argRank
        else:
            self.getField(argField).jRank = argRank

    @error.callStackRoutine
    def getRankAt(self, argField, argRankType):

        if(argRankType not in [RankType.SPRANK, RankType.JRANK]):
            error.LOGGER.report("Invalid argRankType", error.LogType.ERROR)
        
        if(RankType.SPRANK == argRankType):
            return self.getField(argField).spRank
        else:
            return self.getField(argField).jRank

    
    @error.callStackRoutine
    def addAlumnusAt(self, argAlumnusInfo: AlumnusInfo):

        if(self.queryField(argAlumnusInfo.field)):
            return self.getField(argAlumnusInfo.field).addAlumnusWentTo(argAlumnusInfo)
        
        return None

    @error.callStackRoutine
    def getAlumniAt(self, argField, argDestInstitutionId):

        if(self.queryField(argField)):
            return self.getField(argField).getAlumniWentTo(argDestInstitutionId)
        
        return None

    @error.callStackRoutine
    def getNumAlumniAt(self, argField, argDestInstitutionId):

        if(self.queryField(argField)):
            return self.getField(argField).getNumAlumniWentTo(argDestInstitutionId)
        
        return 0

    @error.callStackRoutine    
    def getTotalNumAlumniAt(self, argField):

        if(self.queryField(argField)):
            return self.getField(argField).getTotalNumAlumni()
                
        return 0
    
    @error.callStackRoutine
    def getTotalNumAlumni(self):

        totalNumAlumni = 0

        for field in list(self.fieldDict.values()):

            totalNumAlumni += field.getTotalNumAlumni()

        return totalNumAlumni
    
    @error.callStackRoutine
    def isNonKRInst(self):

        print(f">>> {self.name} {self.country} {int(not util.areSameStrings(self.country, 'KR'))}")
        return int(not util.areSameStrings(self.country, 'KR'))


class InstituionAtField(Institution):

    @error.callStackRoutine
    def __init__(self, argField):

        self.field = argField
        self.spRank = None
        self.jRank = None
        self.alumniDict = {}
        self.alumniDictWithGenderKey = {}

        status.STATTRACKER.statFieldNumInst[self.field] += 1

    @error.callStackRoutine
    def printInfo(self):
        
        print("")
        
        print("=== Field: ", self.field, "===")
        print("SpringRank: ", self.spRank)
        print("Joongang Ranking: ", self.jRank)
        print("Number Of Alumni: ", self.getTotalNumAlumni())

        print("")

    def printAlumniInfo(self):
        for alumniList in util.getValuesListFromDict(self.alumniDict):
            for alumnus in alumniList:
                alumnus.printInfo()

    @error.callStackRoutine
    def addAlumnusWentTo(self, argAlumnusInfo: AlumnusInfo):

        newAlumnus = Alumni(argAlumnusInfo)

        if(argAlumnusInfo.apInstId in self.alumniDict):
            self.alumniDict[argAlumnusInfo.apInstId].append(newAlumnus)
        else:
            self.alumniDict[argAlumnusInfo.apInstId] = [newAlumnus]

        if(argAlumnusInfo.gender in self.alumniDictWithGenderKey):
            self.alumniDictWithGenderKey[argAlumnusInfo.gender].append(newAlumnus)
        else:
            self.alumniDictWithGenderKey[argAlumnusInfo.gender] = [newAlumnus]

    @error.callStackRoutine
    def getAlumniWentTo(self, argDestInstitutionId):
        if(argDestInstitutionId in self.alumniDict):
            return self.alumniDict[argDestInstitutionId]
        else:
            return 0
    
    @error.callStackRoutine
    def getNumAlumniWentTo(self, argDestInstitutionId):
        if(argDestInstitutionId in self.alumniDict):
            return len(self.alumniDict[argDestInstitutionId])
        else:
            return 0
    
    @error.callStackRoutine
    def getTotalNumAlumni(self):

        totalNumAlumni = 0

        for alumniList in self.alumniDict.values():
            totalNumAlumni += len(alumniList)
        
        return totalNumAlumni
    
    @error.callStackRoutine
    def getTotalAlumniList(self):

        returnList = []

        for alumni in util.getValuesListFromDict(self.alumniDict):
            for alumnus in alumni:
                returnList.append(alumnus)


        return returnList

class Alumni:

    @error.callStackRoutine
    def __init__(self, argAlumnusInfo: AlumnusInfo):

        self.field = argAlumnusInfo.field
        self.currentRank = argAlumnusInfo.currentRank
        self.gender = argAlumnusInfo.gender
        self.phDInstId = argAlumnusInfo.phDInstId
        self.phDInstName = argAlumnusInfo.phDInstName
        self.apInstId = argAlumnusInfo.apInstId
        self.apInstName = argAlumnusInfo.apInstName

        self.spRankMove = None
        self.jRankMove = None

        status.STATTRACKER.statTotalNumAlumni+=1

        status.STATTRACKER.statFieldNumAlumni[self.field] += 1

        if(util.Gender.MALE == self.gender):
            status.STATTRACKER.statFieldNumMale[self.field] += 1
        elif(util.Gender.FEMALE == self.gender):
            status.STATTRACKER.statFieldNumFemale[self.field] += 1

        del argAlumnusInfo

    @error.callStackRoutine
    def setRankMove(self, argRankMove, argRankType):

        if(argRankType not in [RankType.SPRANK, RankType.JRANK]):
            error.LOGGER.report("Invalid argRankType", error.LogType.ERROR)
        
        if(RankType.SPRANK == argRankType):
            self.spRankMove = argRankMove
        else:
            self.jRankMove = argRankMove

    @error.callStackRoutine
    def getRankMove(self, argRankType):
        
        if(argRankType not in [RankType.SPRANK, RankType.JRANK]):
            error.LOGGER.report("Invalid argRankType", error.LogType.ERROR)

        if(RankType.SPRANK == argRankType):
            return self.spRankMove
        else:
            return self.jRankMove
    
    @error.callStackRoutine
    def getGender(self):
        return self.gender

    @error.callStackRoutine
    def printInfo(self):

        print("")

        print("---------------------")
        print("Field: ", self.field)
        print("Current Rank: ",self.currentRank)
        print("Gender: ", util.genderToStr(self.gender))
        print("PhD Inst ID: ", self.phDInstId)
        print("PhD Inst Name: ", self.phDInstName)
        print("AP Inst ID: ", self.apInstId)
        print("AP Inst Name: ", self.apInstName)
        print("SpringRank Movement: ", self.spRankMove)
        print("JRank Movement: ", self.jRankMove)
        print("---------------------")



if(__name__ == "__main__"):
    
    error.LOGGER.report("This Module is Not for Main Function", error.LogType.CRITICAL)

