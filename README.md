# FacSIM
Python-based simulator for analyzing hierarchies in faculty hiring network of South Korea

## Directory
### src
Contains source codes.
### dataset
Contains ```instList.xlsx``` (User should manually make this file to initialize instList.), dataset/dirty, and dataset/cleaned.
#### dataset/dirty
Contains dirty datasets, which are not cleaned yet.
### results
Contains directories with results saved, which are classified by simulation date and time.
Results are under the directory named results/YYYY_MM_DD/hh:mm:ss.
#### results/YYYY_MM_DD/hh:mm:ss/cleaned
Contains data cleaned. 
#### results/YYYY_MM_DD/hh:mm:ss/plot
Contains figures generated through ploting methods.
#### results/YYYY_MM_DD/hh:mm:ss/logFile.log
Contains output of ```error.LOGGER```. ```error.LOGGER``` generates log files, seperated in terms of the date.
#### results/YYYY_MM_DD/hh:mm:ss/resultFile.txt
Stdout stream redirected by ```-f``` option is saved here.
#### results/YYYY_MM_DD/hh:mm:ss/setting.ini
Contains configuration file of the certain simulation.
#### results/YYYY_MM_DD/hh:mm:ss/message.txt
(Optional) Contains a simple message about the simulation. Can be designated with -m option.