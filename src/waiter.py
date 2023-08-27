"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""
import datetime as dt
import os
import util
import shutil

global WAITER

class Waiter:
    def __init__(self):
        
        self.currentDateStr = dt.datetime.now().strftime("%Y_%m_%d")
        self.currentTimeStr = dt.datetime.now().strftime("%H:%M:%S")

        self.baseDir = f"../results/{self.currentDateStr}/{self.currentTimeStr}"

        if(not os.path.exists(self.baseDir)):
            os.makedirs(self.baseDir)

        for category in ["plot", "cleaned"]:
            os.makedirs(f"{self.baseDir}/{category}")

        self.__welcome()

    def __welcome(self):

        srcFilePath = "../ini/setting.ini"
        dstFilePath = f"{self.baseDir}/setting.ini"

        if(not os.path.exists(srcFilePath)):
            return 0
        
        return shutil.copy(srcFilePath, dstFilePath)

    def getPlotPath(self, argSubject, argPlotType, argField, *argOthers):

        plotNameList = [argSubject, argPlotType, argField]

        for word in argOthers:
            plotNameList.append(str(word))

        plotName = '_'.join(plotNameList)

        return f"{self.baseDir}/plot/{plotName}.png"

    def getCleanedFilePath(self, argSubject, argField, argFileExt: util.FileExt, *argOthers):

        cleanedFileNameList = [argSubject, argField]

        for word in argOthers:
            cleanedFileNameList.append(str(word))

        cleanedFileName = '_'.join(cleanedFileNameList)
        fileExtStr = util.fileExtToStr(argFileExt)

        cleanedFileDir = f"{self.baseDir}/cleaned/{argField}"

        if(not os.path.exists(cleanedFileDir)):
            os.mkdir(cleanedFileDir)

        return f"{self.baseDir}/cleaned/{argField}/{cleanedFileName}{fileExtStr}"
    
    def getLogFilePath(self):
        return f"{self.baseDir}/logFile.log"

    def getResultFilePath(self):
        return f"{self.baseDir}/resultFile.txt"
    
    def getMessageFilePath(self):
        return f"{self.baseDir}/message.txt"