"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""

import sys
import logging
from enum import Enum
import datetime as dt
import os

global LOGGER

#Function decorator to record function call stack. (Do not modify)
def callStackRoutine(argFunction):
    def wrapper(*args, **kwargs):
        LOGGER.pushCallStack()

        returnValue = argFunction(*args, **kwargs)
        
        LOGGER.popCallStack()
        return returnValue
    return wrapper

class LogType(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class LOGGER_C():
    def __init__(self, argLevel: LogType) -> None:

        fileName = ''.join([dt.datetime.now().strftime("%Y%m%d"), '.log'])
        self.logFilePath = os.path.join('../log', fileName)
        level = None

        
        self.callStack: Stack = Stack()


        match argLevel:
            case LogType.DEBUG:
                level = logging.DEBUG
            case LogType.INFO:
                level = logging.INFO
            case LogType.WARNING:
                level = logging.WARNING
            case LogType.ERROR:
                level = logging.ERROR
            case LogType.CRITICAL:
                level = logging.CRITICAL
            case _:
                currentFunc = sys._getframe().f_code.co_name
                return logging.warning("Invalid Arguments - %s", currentFunc)
        
        logging.basicConfig(filename = self.logFilePath, format = "%(levelname)s: %(message)s", level=level)
        logging.critical(">>> Starting Logging: %s", dt.datetime.now().strftime("%Y%m%d %H:%M:%S"))

    def pushCallStack(self):

        currentFunc = sys._getframe().f_back.f_back.f_code.co_name

        #currentFunc = inspect.stack()[1].function
        self.callStack.push(currentFunc)

        logging.debug("Function Called - %s", currentFunc)


    def popCallStack(self):

        currentFunc = self.callStack.pop()

        logging.debug("Function Returns - %s", currentFunc)

    def __reportDEBUG(self, argMessage):
        currentFunc = sys._getframe().f_back.f_back.f_code.co_name

        return logging.debug("%s - %s", argMessage, currentFunc)

    def __reportINFO(self, argMessage):
        currentFunc = sys._getframe().f_back.f_back.f_code.co_name
        return logging.info("%s - %s", argMessage, currentFunc)

    def __reportWARNING(self, argMessage):
        currentFunc = sys._getframe().f_back.f_back.f_code.co_name
        return logging.warning("%s - %s", argMessage, currentFunc)

    def __reportERROR(self, argMessage):
        currentFunc = sys._getframe().f_back.f_back.f_code.co_name
        return logging.error("%s - %s", argMessage, currentFunc)

    def __reportCRITICAL(self, argMessage):
        currentFunc = sys._getframe().f_back.f_back.f_code.co_name
        logging.critical("%s - %s", argMessage, currentFunc)
        raise Exception(" - ".join([argMessage, currentFunc]))

    #argMessage should be represented in traditional formatting.
    #example: "Function Called - %s", currentFunc (O), "Function Called - " + currentFunc (X)
    def report(self, argMessage: str, argLevel: LogType):

        match argLevel:
            case LogType.DEBUG:
                return self.__reportDEBUG(argMessage)
            case LogType.INFO:
                return self.__reportINFO(argMessage)
            case LogType.WARNING:
                return self.__reportWARNING(argMessage)
            case LogType.ERROR:
                return self.__reportERROR(argMessage)
            case LogType.CRITICAL:
                return self.__reportCRITICAL(argMessage)
            case _:
                currentFunc = sys._getframe().f_code.co_name
                return logging.warning("Invalid Arguments - %s", currentFunc)




#[IMPORTANT] since class Stack itself is resonsible for recording function call stack,
#methods of class Stack are not decoreated with callStackRoutine()
#decorating methods of class Stack with callStackRoutine() brings RecursionError.
class Stack:

    def __init__(self):
        self.items=[]
    
    def pop(self):
        stackLength = len(self.items)
        
        if stackLength < 1:
            return printErrorMessage("Stack is Empty")
            
        result = self.items[stackLength-1]
        del self.items[stackLength-1]
        return result
    

    def push(self,x):
        self.items.append(x)
    
    def peek(self):
    	return self.items[-1]

    def isEmpty(self):
        return not self.items

@callStackRoutine
def printErrorMessage(argMessage):

    #currentFunc = inspect.stack()[1].function
    currentFunc = sys._getframe().f_back.f_back.f_code.co_name
    print(currentFunc, ":", argMessage, '\n')

    return 0

@callStackRoutine
def printErrorMessageAndExit(argMessage):
    #currentFunc = inspect.stack()[1].function
    currentFunc = sys._getframe().f_back.f_back.f_code.co_name
    print(currentFunc, ":", argMessage, '\n')
    sys.exit()


if(__name__ == '__main__'):

    raise Exception("This Module is Not for Main Function")