"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""

import util
import error

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
        instLocation:list = argInstLocationInfo.split(',')
        
        if(3 != len(instLocation)):
            error.LOGGER.report("Invalid Institution Infof may Yield in Wrong Results", error.LogType.WARNING)
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
        if(util.isEmptyData(self.instId)):
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
    

class Institution:

    @error.callStackRoutine
    def __init__(self, argInstInfo: InstInfo):

        self.instId = argInstInfo.instId
        self.name = argInstInfo.name
        self.region = argInstInfo.region
        self.country = argInstInfo.country
        self.fieldDict = {}

        del argInstInfo

    @error.callStackRoutine
    def printInfo(self):

        print("===========================================================")

        print("Institution Id: ", self.instId)
        print("Name: ", self.name)
        print("Region: ", self.region)
        print("Country: ", self.country)
        print("")

        for item in sorted(self.fieldDict.items()):

            department = self.fieldDict[item[0]]
            department.printInfo()


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
    def setMVRRankAt(self, argField, argMVRRank):

        self.getField(argField).MVRRank = argMVRRank

    @error.callStackRoutine
    def getMVRRankAt(self, argField):

        return self.getField(argField).MVRRank
    
    @error.callStackRoutine
    def addAlumnusAt(self, argField, argDestInstitutionId, argCurrentRank, argGender):

        if(self.queryField(argField)):
            return self.getField(argField).addAlumnusWentTo(argDestInstitutionId, argCurrentRank, argGender)
        
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


class InstituionAtField(Institution):

    @error.callStackRoutine
    def __init__(self, argField):

        self.field = argField
        self.MVRRank = None
        self.alumniDict = {}

    @error.callStackRoutine
    def printInfo(self):
        
        print("=== Field: ", self.field, "===")
        print("MVR Ranking: ", self.MVRRank)
        print("Number Of Alumni: ", self.getTotalNumAlumni())

        print("")

    @error.callStackRoutine
    def addAlumnusWentTo(self, argDestInstitutionId, argCurrentRank, argGender):

        newAlumnus = Alumni(argCurrentRank, argGender)

        if(argDestInstitutionId in self.alumniDict):
            self.alumniDict[argDestInstitutionId].append(newAlumnus)
        else:
            self.alumniDict[argDestInstitutionId] = [newAlumnus]

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




class Alumni:

    @error.callStackRoutine
    def __init__(self, argCurrentRank, argGender):

        self.currentRank = argCurrentRank
        self.gender = argGender



if(__name__ == "__main__"):
    
    error.LOGGER.report("This Module is Not for Main Function", error.LogType.CRITICAL)

