import configparser
import os
import error
import career
import util

global PARAM

class Setting:
    def __init__(self):

        configPath = "../ini/setting.ini"

        if(not os.path.exists(configPath)):
            error.LOGGER.report("No setting.ini", error.LogType.CRITICAL)

        self.config = configparser.ConfigParser(allow_no_value = True)
        self.config.optionxform = str
        self.config.read(configPath, encoding='utf-8')

        self.param = {}

    def getParam(self):
        return self.param

    def parse(self):

        if(self.__parseBasic()):
            error.LOGGER.report("Parsing Basic Section: Done", error.LogType.INFO)

            if(self.__parseRankType()):
                error.LOGGER.report("Parsing RankType Section: Done", error.LogType.INFO)

                if(self.__parseAnalysis()):
                    error.LOGGER.report("Parsing Analysis Section: Done", error.LogType.INFO)
                    return 1
        return 0

    def __parseBasic(self):

        section = "Basic"

        self.param[section] = {}

        if(not self.config.has_section(section)):
            return 0

        # targetDeg
        if(self.config.has_option(section, "targetDeg")):
            self.param[section]["targetDeg"] = []
            values = self.config.get(section, "targetDeg")

            for value in values.split(', '):
                if(not util.isLabelOfEnum(career.DegreeType, value)):
                    return 0
                
                self.param[section]["targetDeg"].append(career.DegreeType[value])
                
        else:
            return 0
        
        # networkType
        if(self.config.has_option(section, "networkType")):
            self.param[section]["networkType"] = None
            value = self.config.get(section, "networkType")

            if(value not in ["CLOSED", "OPENED"]):
                return 1
            
            self.param[section]["networkType"] = value

        else:
            return 0

        return 1

    def __parseRankType(self):
        
        section = "RankType"
        self.param[section] = {}

        if(not self.config.has_section(section)):
            return 0

        degList = self.config.options(section)

        for deg in degList:
            if(not util.isLabelOfEnum(career.DegreeType, deg)):
                return 0
            
            if(self.config.has_option(section, deg)):
                self.param[section][career.DegreeType[deg]] = None
                value = self.config.get(section, deg)

                parenIndex1 = value.find("(")
                parenIndex2 = value.find(")")

                rankTypeStr = value[:parenIndex1]

                if(rankTypeStr not in ["SpringRank", "Custom"]):
                    return 0
                
                descripStr = value[parenIndex1+1:parenIndex2]

                match rankTypeStr:
                    case "SpringRank":
                        degTypeList = descripStr.split("->")

                        if(2!=len(degTypeList)):
                            return 0

                        for degType in degTypeList:
                            if(not util.isLabelOfEnum(career.DegreeType, degType)):
                                return 0
                            
                        srcDegType = degTypeList[0]
                        dstDegType = degTypeList[1]

                        if(career.DegreeType[srcDegType] not in self.param["Basic"]["targetDeg"]):
                            return 0
                        
                        if(career.DegreeType[dstDegType] not in self.param["Basic"]["targetDeg"]):
                            return 0
                            
                        self.param[section][career.DegreeType[deg]] = (rankTypeStr, (career.DegreeType[srcDegType], career.DegreeType[dstDegType]))

                    case "Custom":
                        filePathStr = descripStr
                        #validation required
                        self.param[section][career.DegreeType[deg]] = (rankTypeStr, filePathStr)

            else:
                return 0

        return 1

    def __parseAnalysis(self):
        return 1

