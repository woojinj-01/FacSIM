from enum import Enum
import error
import institution
import util
import status

class DegreeType(Enum):

    BS = 0
    MS = 1
    PHD = 2
    PDOC = 3
    AP = 4

    def toStr(self, argPurpose):

        if("search" == argPurpose):

            match self:
                case DegreeType.BS:
                    return "BS"
                
                case DegreeType.MS:
                    return "MS"
                
                case DegreeType.PHD:
                    return "PHD"
                
                case DegreeType.PDOC:
                    return "PDOC"
                
                case DegreeType.AP:
                    return "Assistant Professor"
                
        elif("label" == argPurpose):
            return self.name

        else:
            return None

    def next(self):
        targetIndex = self.value + 1

        if(targetIndex < len(DegreeType)):
            return DegreeType(targetIndex)
        else:
            return None

class CareerStep:

    def __init__(self, argInst, argDegType):

        self.inst: institution.Institution = argInst
        self.degType: DegreeType = argDegType

    def isComplete(self):
        return isinstance(self.inst, institution.Institution)\
            and isinstance(self.degType, DegreeType)
    
    def __repr__(self) -> str:

        if(not self.isComplete()):
            return ''

        strList = [f'\n========{self.degType}========', \
                    f'Inst ID: {self.inst.instId}', \
                    f'Inst Name: {self.inst.name}', \
                    f'==============================\n']
        
        return '\n'.join(strList)

class Career:

    def __init__(self):

        self.careerPath = []
        self.pointer = 0

    def append(self, argCareerStep: CareerStep):

        if(not argCareerStep.isComplete()):
            
            error.LOGGER.report("Cannot Append Empty CareerStep", error.LogType.ERROR)
            return 0
        
        elif(self.query(argCareerStep.degType)):

            error.LOGGER.report("This Step Already Exists.", error.LogType.WARNING)
            return 0
        
        self.careerPath.append(argCareerStep)

        self.careerPath.sort(key=lambda x: x.degType.value)

        return 1
    
    def query(self, argDegType):

        for step in self.careerPath:
            if(argDegType == step.degType):
                return step
            
        return None

    def __iter__(self):
        return self
    
    def __next__(self):

        if(self.pointer < len(self.careerPath)):
            
            returnValue = self.careerPath[self.pointer]
            self.pointer += 1

            return returnValue
        
        else:
            self.pointer = 0

            raise StopIteration
        
class Alumni:

    @error.callStackRoutine
    def __init__(self, argField, argCurrRank, argGender: util.Gender, argPID):

        self.field = argField
        self.currentRank = argCurrRank
        self.gender = argGender
        self.pid = argPID

        self.career = Career()

        self.spRankMove = None
        self.jRankMove = None

        status.STATTRACKER.statTotalNumAlumni+=1

        status.STATTRACKER.statFieldNumAlumni[self.field] += 1

        if(util.Gender.MALE == self.gender):
            status.STATTRACKER.statFieldNumMale[self.field] += 1
        elif(util.Gender.FEMALE == self.gender):
            status.STATTRACKER.statFieldNumFemale[self.field] += 1

    @error.callStackRoutine
    def __repr__(self) -> str:
        
        strList = []

        strList.append("")
        strList.append("---------------------")
        strList.append(f"Field: {self.field}")
        strList.append(f"Current Rank: {self.currentRank}")
        strList.append(f"Gender: {util.genderToStr(self.gender)}")
        strList.append(f"PID: {self.pid}")

        for step in self.career:
            strList.append(step.__repr__())

        strList.append("---------------------")

        return '\n'.join(strList)

    @error.callStackRoutine
    def getRankMove(self, argDegTuple):

        srcDegType = argDegTuple[0]
        dstDegType = argDegTuple[1]

        srcInst = self.getInstFor(srcDegType)
        dstInst = self.getInstFor(dstDegType)

        if(None == srcInst or None == dstInst):
            return None
        
        srcInstRank = srcInst.getRankAt(self.field, srcDegType)
        dstInstRank = dstInst.getRankAt(self.field, dstDegType)

        if(None == srcInstRank or None == dstInstRank):
            return None
        
        return (dstInstRank - srcInstRank)
    
    @error.callStackRoutine
    def getGender(self):
        return self.gender
    
    @error.callStackRoutine
    def append(self, argCareerStep: CareerStep):
        returnValue = self.career.append(argCareerStep)

        if(1 == returnValue):
            argCareerStep.inst.addAlumnus(self, argCareerStep.degType)

        return returnValue
    
    @error.callStackRoutine
    def query(self, argDegType: DegreeType):
        return self.career.query(argDegType)
    
    @error.callStackRoutine
    def getInstFor(self, argDegType: DegreeType):

        step = self.query(argDegType)

        if(None == step):
            return None
        else:
            return step.inst