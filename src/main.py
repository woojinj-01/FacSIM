"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""

import analyzer
import error
import argparse
import util
import sys

#not tracked by callstack routine
def parseOptions():

    returnDict = {}
    parser = argparse.ArgumentParser(description='Add options for further functionalities.')

    parser.add_argument('-l', '--log', choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],\
                        help='Set the logging threshold. 0 is the lowest, 4 is the highest.', default='WARNING')
    
    parser.add_argument('-f', '--file', action = 'store_true',\
                         help='Redirect stdout stream to result.txt', default= False)

    args = parser.parse_args()

    
    
    return args

#not tracked by callstack routine
def parseOptionsAndInit():
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

    if(args.file):
        sys.stdout = open('../results.txt', 'w')

if(__name__ == '__main__'):

    parseOptionsAndInit()

    analyzer = analyzer.Analyzer()
    analyzer.loadInstIdDictFrom("../dataset/instList.xlsx")

    analyzer.cleanData()
    analyzer.exportVertexAndEdgeListForAll(util.FileExt.CSV)

    analyzer.calcMVRRAnkForAll()

    util.callAndPrint(analyzer.calcGiniCoeffForAll)()

    util.callAndPrint(analyzer.calcAvgMVRMoveBasedOnGender)(util.Gender.MALE)
    util.callAndPrint(analyzer.calcAvgMVRMoveBasedOnGender)(util.Gender.FEMALE)

    util.callAndPrint(analyzer.calcAvgMVRMoveForRange)(0, 15)
    util.callAndPrint(analyzer.calcAvgMVRMoveForRange)(15, 100)
    

    