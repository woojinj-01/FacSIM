"""
Author: Woojin Jung (GitHub: woojinj-01)
Email: blankspace@kaist.ac.kr

"""
global STATTRACKER

class StatusTracker:

    def __init__(self):
        
        #Status (list) total
        self.statInstNameList = []
        self.statRegionList = []
        self.statCountryList = []

        # Status (numbers) total
        self.statTotalNumRowCleaned = 0
        self.statTotalNumInst = 0
        self.statTotalNumAlumni = 0
        self.statTotalNumMale = 0
        self.statTotalNumFemale = 0

        # Status (numbers) per field
        self.statFieldNumInst = {'Physics': 0, 'Computer Science': 0}
        self.statFieldNumAlumni = {'Physics': 0, 'Computer Science': 0}
        self.statFieldNumMale = {'Physics': 0, 'Computer Science': 0}
        self.statFieldNumFemale = {'Physics': 0, 'Computer Science': 0}

        # Missing datas per field
        self.statNumRowPhDWithNoAP = {'Physics': 0, 'Computer Science': 0}
        self.statNumRowAPWithNoPhD = {'Physics': 0, 'Computer Science': 0}
        self.statNumRowWithNoPhDAndAP = {'Physics': 0, 'Computer Science': 0}


    def print(self):

        members = vars(self)

        for name, value in members.items():
            print(name, ":", value)



