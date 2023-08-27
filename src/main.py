"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""

import analyzer
import error
import argparse
import util
import sys
import status
import setting
from career import DegreeType
import waiter

StatusTracker = status.StatusTracker()

#not tracked by callstack routine
def parseOptions():

    parser = argparse.ArgumentParser(description='Options for further functionalities.')

    parser.add_argument('-l', '--log', choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],\
                        help='Set the logging threshold. 0 is the lowest, 4 is the highest.', default='WARNING')
    
    parser.add_argument('-f', '--file', action = 'store_true',\
                         help='Redirect stdout stream to result.txt, except for user interactive message.', default= False)

    parser.add_argument('-c', '--correction', choices = ['ENABLE', 'DISABLE'], \
                        help= 'Enables user-interactive typo correction. Disable this mode for faster, but coarser analysis.\
                            Strongly recommended to be abled when precise results are required. Note that non-interactive typo corrections \
                                will be still applied even if the mode is turned off.', default='ENABLE')
    
    parser.add_argument('-m', '--message', type = str,\
                         help='Specify a message that will be stored in the results folder,\
                            in order to provide more description for the trial.', default= False)
    
    args = parser.parse_args()

    
    
    return args

#not tracked by callstack routine
def init():

    args = parseOptions()

    match args.log:
        case 'DEBUG':
            logType = error.LogType.DEBUG
        case 'INFO':
            logType = error.LogType.INFO
        case 'WARNING':
            logType = error.LogType.WARNING
        case 'ERROR':
            logType = error.LogType.ERROR
        case 'CRITICAL':
            logType = error.LogType.CRITICAL
        case _:
            logType = None

    waiter.WAITER = waiter.Waiter()
    error.LOGGER = error.LOGGER_C(logType, waiter.WAITER.getLogFilePath())

    settingParser = setting.Setting()
    properSetting = settingParser.parse()
    
    if(not properSetting):
        return 0

    setting.PARAM = settingParser.getParam()

    del settingParser

    status.STATTRACKER = status.StatusTracker()
    util.TYPOHISTORY = util.TypoHistory(args.correction)

    if(args.file):
        resultFilePath = waiter.WAITER.getResultFilePath()
        sys.stdout = open(resultFilePath, 'w')

    if(args.message):
        
        messageFilePath = waiter.WAITER.getMessageFilePath()

        with open(messageFilePath, "w") as file:
            file.write(args.message)

    return args

if(__name__ == '__main__'):

    args = init()

    if(0 == args):
        exit()
    
    print(setting.PARAM)

    analyzer = analyzer.Analyzer()
    analyzer.loadInstIdDictFrom("../dataset/instList.xlsx")

    analyzer.cleanData()
    analyzer.exportVertexAndEdgeList(util.FileExt.CSV)

    analyzer.setRanks()

    analyzer.exportRankList(DegreeType.PHD, util.FileExt.CSV)

    #analyzer.plotLorentzCurveIntegrated((DegreeType.PHD, DegreeType.AP))
    
    #analyzer.plotRankMove((0, 20), (DegreeType.PHD, DegreeType.AP))
    #analyzer.plotRankMove((20, 40), (DegreeType.PHD, DegreeType.AP))
    #analyzer.plotRankMove((40, 60), (DegreeType.PHD, DegreeType.AP))
    #analyzer.plotRankMove((60, 80), (DegreeType.PHD, DegreeType.AP))
    #analyzer.plotRankMove((80, 100), (DegreeType.PHD, DegreeType.AP))

    #analyzer.plotRankMove((0, 100), (DegreeType.PHD, DegreeType.AP))

    #analyzer.plotGenderRatio()

    #analyzer.plotNonKR((DegreeType.PHD, DegreeType.AP), False, 1)
    #analyzer.plotNonKR((DegreeType.PHD, DegreeType.AP), False, 10)
    #analyzer.plotNonKR((DegreeType.BS, DegreeType.AP), True, 20)
    #analyzer.plotNonKR((DegreeType.PHD, DegreeType.AP), False)
    #analyzer.plotNonKRFac(career.DegreeType.BS)
    #analyzer.plotNonKRFacRatio()

    #analyzer.printAllInstitutions(granularity = "alumni")

    status.STATTRACKER.print()

    util.TYPOHISTORY.flush()    #mandatory
    

    