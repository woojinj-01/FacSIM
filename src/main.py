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
import institution

StatusTracker = status.StatusTracker()

#not tracked by callstack routine
def parseOptions():

    returnDict = {}
    parser = argparse.ArgumentParser(description='Options for further functionalities.')

    parser.add_argument('-l', '--log', choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],\
                        help='Set the logging threshold. 0 is the lowest, 4 is the highest.', default='WARNING')
    
    parser.add_argument('-f', '--file', action = 'store_true',\
                         help='Redirect stdout stream to result.txt, except for user interactive message.', default= False)
    
    parser.add_argument('-c', '--correction', action = 'store_true', \
                        help= 'Enables user-interactive typo correction. Disable this mode for faster, but coarser analysis.\
                            Strongly recommended to be abled when precise results are required. Note that non-interactive typo corrections \
                                will be still applied even if the mode is turned off.')
    
    parser.add_argument('-j', '--joongang', action = 'store_true', \
                        help= 'Takes Joongang-ilbo University Rank into consideration. \
                            With this option, All analysis will target both SpringRank and Joongang-ilbo Rank.')

    args = parser.parse_args()

    
    
    return args

#not tracked by callstack routine
def init():
    #TODO: Should implement Joongang Rank Part

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

    error.LOGGER = error.LOGGER_C(logType)
    status.STATTRACKER = status.StatusTracker()
    util.TYPOHISTORY = util.TypoHistory(args.correction)

    if(args.file):
        resultFilePath = util.getResultFilePath(util.FileExt.TXT)
        sys.stdout = open(resultFilePath, 'w')

if(__name__ == '__main__'):

    init()

    targetRankList= [institution.RankType.SPRANK, institution.RankType.JRANK]

    analyzer = analyzer.Analyzer(targetRankList)
    analyzer.loadInstIdDictFrom("../dataset/instList.xlsx")

    analyzer.cleanData()
    analyzer.exportVertexAndEdgeListForAll(util.FileExt.CSV)

    analyzer.calcRanksForAll()

    #analyzer.calcMVRRankMoveForAll()

    #analyzer.calcGiniCoeffForAll()
    
    #analyzer.plotRankMove(0, 20)
    #analyzer.plotRankMove(20, 40)
    #analyzer.plotRankMove(40, 60)
    #analyzer.plotRankMove(60, 80)
    #analyzer.plotRankMove(80, 100)

    #analyzer.plotRankMove(0, 100)

    #analyzer.plotGenderRatio()

    analyzer.plotNonKRFac()
    analyzer.plotNonKRFacRatio()

    #analyzer.printAllInstitutions(granularity = "field")

    status.STATTRACKER.print()

    util.TYPOHISTORY.flush()    #mandatory
    

    