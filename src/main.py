"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""

import analyzer
import error

if(__name__ == '__main__'):

    error.LOGGER = error.LOGGER_C(error.LogType.WARNING)

    analyzer = analyzer.Analyzer()
    analyzer.loadInstIdDictFrom("../dataset/instList.xlsx")
    
    analyzer.cleanData()
    analyzer.exportVertexAndEdgeListForAll(".csv")

    print(analyzer.calcGiniCoeffForAll())
    analyzer.calcMVRRAnkForAll()
    analyzer.printAllInstitutions()