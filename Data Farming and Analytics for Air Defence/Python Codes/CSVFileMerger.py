# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 20:07:21 2022

@author: Arshad Mehtiyev
"""

import pandas as pd
from tkinter import Tk, filedialog
import os



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

def extractCSVFiles(listOfFiles):
    csvFilesNameList = list()
    csvFilesList = list()
    for file in listOfFiles:
        if os.path.splitext(file)[1] == '.csv':
            csvFilesNameList += [os.path.basename(os.path.splitext(file)[0])]
            csvFilesList +=[file] 
    print('\nThere are ' + str(len(csvFilesNameList)) + ' CSV files\n')
    return csvFilesNameList, csvFilesList 




def main():
    # Get the path to directory of files
    dir_path = getDirectory()
    
    # Ask where to save final result in CSV format
    savePath = saveFile()
    
    # Get the list of all files(with their path) in the direcotry
    listOfFiles = filesList(dir_path)
    
    # Make the list of csv files located in the directory
    csvFilesNameList, csvFilesList = extractCSVFiles(listOfFiles)
    
    #combining all of the files in the list
    dataFrameList =[pd.read_csv(file) for file in csvFilesList]
    combinedFinalData = pd.concat([file for file in dataFrameList])
    
    # exporting csv file
    combinedFinalData.to_csv(savePath, index=False, line_terminator='\n')


if __name__ == '__main__':
    main()