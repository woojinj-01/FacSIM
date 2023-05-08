"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""

import analyzer
import error
import argparse

#not tracked by callstack routine
def parseOptions():

    returnDict = {}
    parser = argparse.ArgumentParser(description='Add options for further functionalities.')
    parser.add_argument('-l', '--log', choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging threshold. 0 is the lowest, 4 is the highest.', default='WARNING')

    args = parser.parse_args()

    match args.log:
        case 'DEBUG':
            returnDict['log'] = error.LogType.DEBUG
        case 'INFO':
            returnDict['log'] = error.LogType.INFO
        case 'WARNING':
            returnDict['log'] = error.LogType.WARNING
        case 'ERROR':
            returnDict['log'] = error.LogType.ERROR
        case 'CRITICAL':
            returnDict['log'] = error.LogType.CRITICAL
        case _:
            returnDict['log'] = None

    

    return returnDict

def parseOptionsAndInit():
    args = parseOptions()
    error.LOGGER = error.LOGGER_C(args['log'])

if(__name__ == '__main__'):

    parseOptionsAndInit()

    analyzer = analyzer.Analyzer()
    analyzer.loadInstIdDictFrom("../dataset/instList.xlsx")
    
    analyzer.cleanData()