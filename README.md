# FacSIM
Python-based simulator for analyzing hierarchies in faculty hiring network of South Korea

## Directory
### src
Contains source codes.
### dataset
Contains ```instList.xlsx``` (User should manually make this file to initialize instList.), dataset/dirty, and dataset/cleaned.
#### dataset/dirty
Contains dirty datasets, which are not cleaned yet.
#### dataset/cleaned
Contains output of ```analyzer.Analyzer```, which are datasets cleaned.
#### dataset/jrank
Contains joongang-ilbo rank datasets, which are crawled from https://namu.wiki/w/중앙일보%20대학평가#s-3.
### log
Contains output of ```error.LOGGER```. ```error.LOGGER``` generates log files, seperated in terms of the date.
### result
Stdout stream redirected by ```-f``` option is saved here.
### ini
Contains ```globalTypoHistory.ini``` (automatically generated).
### plot
Contains figures generated through ploting methods.