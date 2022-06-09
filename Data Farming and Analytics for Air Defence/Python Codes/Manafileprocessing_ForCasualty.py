# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:14:22 2022

@author: Arshad Mehtiyev
"""

# importing the required modules

import pandas as pd
from tkinter import Tk, filedialog
import os
from xml.dom import minidom



def getDirectory():
    root = Tk() # pointing root to Tk() to use it as Tk() in program.
    root.withdraw() # Hides small tkinter window.
    root.attributes('-topmost', True) # Opened windows will be active. above all windows despite of selection.
    dir_path = filedialog.askdirectory(title='Select folder containing the files and subfolders') # Returns opened path as str
    print("\nSelected directory is:\n", dir_path)
    return dir_path

def saveFile():
    root = Tk() # pointing root to Tk() to use it as Tk() in program.
    root.withdraw() # Hides small tkinter window.
    root.attributes('-topmost', True) # Opened windows will be active. above all windows despite of selection.
    types=[('CSV types(*.csv)','*.csv')]
    save_path = filedialog.asksaveasfile(title='Where to save(Specify file name)',
                                         filetypes = types, 
                                         defaultextension = types) # Returns opened path as str
    print("\nSelected folder to save is:\n", save_path.name)
    return save_path.name

def filesList(directory): 
    
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(directory):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    print("\nThere are "+str(len(listOfFiles))+ " files in the directory")
    return listOfFiles

def extractXMLFiles(listOfFiles):
    xmlFilesNameList = list()
    xmlFilesList = list()
    for file in listOfFiles:
        if os.path.splitext(file)[1] == '.xml':
            xmlFilesNameList += [os.path.basename(os.path.splitext(file)[0])]
            xmlFilesList +=[file] 
    print('\nThere are ' + str(len(xmlFilesNameList)) + ' XML files\n')
    return xmlFilesNameList, xmlFilesList


def extractCSVFiles(listOfFiles):
    csvFilesNameList = list()
    csvFilesList = list()
    for file in listOfFiles:
        if os.path.splitext(file)[1] == '.csv':
            csvFilesNameList += [os.path.basename(os.path.splitext(file)[0])]
            csvFilesList +=[file] 
    print('\nThere are ' + str(len(csvFilesNameList)) + ' CSV files\n')
    return csvFilesNameList, csvFilesList 

def getFileName(file):
    return str(os.path.basename(os.path.splitext(file)[0]))
    
def processCSV(file):
    #open the file
    infile = open(file, "r")
    
    # read content
    content = infile.readlines()
    infile.close()
    
    new_content=[]
        
    # Getting the name of the file
    ##filename= ((content[3]).split('\\')[-1]).strip()
    
    # Getting only the required lines from the output file
    for line in content[5:16]:
        strippedLine=line.strip()
        new_content.append(strippedLine)
    
    # Making DataFrame from the cleaned output data
    data = pd.DataFrame([sub.split(",") for sub in new_content])    
    new_header = data.iloc[0] #grab the first row for the header
    data = data[1:] #take the data less the header row
    data.columns = new_header #set the header row as the df header
    
    # Adding filename(Scenario) name as a new feature 
    data['Scenario Name']=getFileName(file)
    
    return data


def getCasualty(mainFile, csvFilesList):
    fileName = getFileName(mainFile)
    
    missilesDestroyed = [0]*10
    
    #rawDataList=[]
    
    for casFile in csvFilesList:
        new_content=[]
        
        if fileName + '_' in getFileName(casFile):

            infile = open(casFile, "r")
            # read content
            content = infile.readlines()
            infile.close()

            index = int(getFileName(casFile).split('_')[-1])

            # Getting the name of the file
            #filename= ((content[3]).split('\\')[-1]).strip()

            # Getting only the required lines from the output file

            for line in content[5:len(content)-1]:
                strippedLine=line.strip()
                new_content.append(strippedLine)


            rawData = pd.DataFrame([sub.split(",") for sub in new_content])
            new_header = rawData.iloc[0] #grab the first row for the header
            rawData = rawData[1:] #take the data less the header row
            rawData.columns = new_header #set the header row as the df header

            #rawDataList[index-1]=rawData

            missilesDestroyed[index-1]=len((rawData.loc[(rawData['squad']=='Red') & 
                                           (rawData['x']<='390')]).index)

    return missilesDestroyed

def readXML(file, data):
    
    XMLObj = minidom.parse(file)
    
    # Missile Quantity
    missileQTY=0

    for squad in XMLObj.getElementsByTagName('Squad'):
        for index in squad.getElementsByTagName('index'):
            if int(index.firstChild.data) in [10,11,12]:
                for agents in squad.getElementsByTagName('NumAgents'):
                    missileQTY+=int(agents.firstChild.data)
    
    # Stealth number for each missile class(10,11,12)
    
    missileStealth=[]

    for squad in XMLObj.getElementsByTagName('Squad'):
        for index in squad.getElementsByTagName('index'):
            if int(index.firstChild.data) in [10,11,12]:
                for state in squad.getElementsByTagName('state'):
                    for stealth in state.getElementsByTagName('Stealth'):
                        missileStealth.append(int(stealth.firstChild.data))
    missileClass_10_Stealth = missileStealth[0]
    missileClass_11_Stealth = missileStealth[1]
    missileClass_12_Stealth = missileStealth[2]
    
    # Asset quantity
    
    assetQTY=0

    for squad in XMLObj.getElementsByTagName('Squad'):
        for index in squad.getElementsByTagName('index'):
            if int(index.firstChild.data) in [1,2,3]:
                for agents in squad.getElementsByTagName('NumAgents'):
                    assetQTY+=int(agents.firstChild.data)
    
    
    # Interceptor Quantity
    interceptorQTY=0

    for squad in XMLObj.getElementsByTagName('Squad'):
        for index in squad.getElementsByTagName('index'):
            if int(index.firstChild.data) in [7,8,9]:
                for agents in squad.getElementsByTagName('NumAgents'):
                    interceptorQTY+=int(agents.firstChild.data)
    
    #Interceptor Sensor Range
    interceptorSensorRange=0

    for squad in XMLObj.getElementsByTagName('Squad'):
        for index in squad.getElementsByTagName('index'):
            if int(index.firstChild.data)==7:
                for state in squad.getElementsByTagName('state'):
                    for statename in state.getElementsByTagName('StateName'):
                        if statename.firstChild.data == ' Default State ': 
                            for sensorstate in state.getElementsByTagName('SensorState'):
                                for sensorstate in state.getElementsByTagName('SensorState'):
                                    for sensstclass in sensorstate.getElementsByTagName('SensStClass'):
                                        interceptorSensorRange = int(sensstclass.firstChild.data)

    # Interceptor Movement Speed
    interceptorMovementSpeed=0

    for squad in XMLObj.getElementsByTagName('Squad'):
        for index in squad.getElementsByTagName('index'):
            if int(index.firstChild.data)==7:
                for state in squad.getElementsByTagName('state'):
                    for statename in state.getElementsByTagName('StateName'):
                        if statename.firstChild.data == ' Contact State ': 
                            for ranges in state.getElementsByTagName('range'):
                                for rangename in ranges.getElementsByTagName('RangeName'):
                                    if rangename.firstChild.data == ' Movement Speed   ':
                                        for rangevalue in ranges.getElementsByTagName('RangeVal'):
                                            interceptorMovementSpeed = int(rangevalue.firstChild.data)
      
    # Interceptor Hit probability
    interceptorHitProbability=0

    for squad in XMLObj.getElementsByTagName('Squad'):
        for index in squad.getElementsByTagName('index'):
            if int(index.firstChild.data)==7:
                for state in squad.getElementsByTagName('state'):
                    for statename in state.getElementsByTagName('StateName'):
                        if statename.firstChild.data == ' Contact State ': 
                            for weaponstate in state.getElementsByTagName('WeaponState'):
                                for sskptable in weaponstate.getElementsByTagName('sskpTable'):
                                    for sskptablepoint in sskptable.getElementsByTagName('sskpTablePoint'): 
                                        for sskptableprob in sskptablepoint.getElementsByTagName('sskpTableProb'): 
                                            interceptorHitProbability=float(sskptableprob.firstChild.data)/10000
 
    data['Asset Amount']= assetQTY
    data['Missile Quantity'] = missileQTY
    data['Missile Class_10 Stealth'] = missileClass_10_Stealth
    data['Missile Class_11 Stealth'] = missileClass_11_Stealth
    data['Missile Class_12 Stealth'] = missileClass_12_Stealth
    data['Interceptor Quantity'] = interceptorQTY
    data['Interceptor Sensor Range'] = interceptorSensorRange
    data['Interceptor Speed'] = interceptorMovementSpeed
    data['Interceptor Hit Probability'] = interceptorHitProbability                              
        
    return data    


def main():
    
    dataFrameList =list()
    
    # Get the path to directory of files
    dir_path = getDirectory()
    
    # Ask where to save final result in CSV format
    savePath = saveFile()
    
    # Get the list of all files(with their path) in the direcotry
    listOfFiles = filesList(dir_path)
    
    # Extract the list of XML file names from listOfFiles
    xmlFilesNameList, xmlFilesList = extractXMLFiles(listOfFiles)
    
    # Make the list of csv files located in the directory
    csvFilesNameList, csvFilesList = extractCSVFiles(listOfFiles)
    
    
    #process counter
    pCntr=0 
    step = len(xmlFilesList)//5
    # Check if csv file has matching XML file 
    for file in csvFilesList:
        if (getFileName(file) in xmlFilesNameList and 
            len(getFileName(file))>9):
            pCntr+=1
    
            # Process the CSV file, convert to DataFrame and add filename as column
            data = processCSV(file)
            
            # Adding new column with amount of missiles sucessfully destroyed by interceptors
            
            data['SucessfullyDestroyedMissiles']=getCasualty(file, csvFilesList)
                   
            
            # Read respective XML file and add required information to the dataframe
            index = xmlFilesNameList.index(getFileName(file))
            data = readXML(xmlFilesList[index],data)
    
            # Add dataframe to the list
            dataFrameList.append(data)
            if pCntr==step:
                print('Processed ' + str(pCntr)+ ' out of ' + str(len(xmlFilesList)))
                step+=step
    #print(len(dataFrameList))
    
    # Stacking (merging) all dataframes    
    finalDataFrame = pd.concat(dataFrameList)
    
    #list of columns to drop
    columnsToDrop=[' Sqd1Inj',' Sqd2Inj',' Sqd3Inj',' Sqd4Inj',' Sqd5Inj',
                   ' Sqd6Inj',' Sqd7Inj',' Sqd8Inj',' Sqd9Inj',' Sqd10Inj',
                   ' Sqd11Inj',' Sqd12Inj']
    
    
    #dropping not required columns
    finalDataFrame = finalDataFrame.drop(columns=columnsToDrop)
    
    # adding new column "AssetSurvivalRate"
    sqd1Cas = pd.to_numeric(finalDataFrame[' Sqd1Cas'])
    sqd2Cas = pd.to_numeric(finalDataFrame[' Sqd2Cas'])
    sqd3Cas = pd.to_numeric(finalDataFrame[' Sqd3Cas'])
    totalAssetCas = sqd1Cas +sqd2Cas +sqd3Cas                                                     
    
    finalDataFrame['AssetSurvivalRate'] = round(1-totalAssetCas/pd.to_numeric(finalDataFrame['Asset Amount']),2)
    
    # adding new column "DefenseSuccessRate"
    finalDataFrame['DefenseSuccessRate'] = round(pd.to_numeric(finalDataFrame['SucessfullyDestroyedMissiles'])/
                                                 pd.to_numeric(finalDataFrame['Missile Quantity']),3)
    
    
    
    # Convert final dataframe to CSV and export
    
    
    finalDataFrame.to_csv(savePath, index=False, line_terminator='\n')
    
    print("Total of " + str(pCntr) + " csv files were processed")


if __name__ == '__main__':
    main()


