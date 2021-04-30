import os

def ParseArgsForBatch(batchPath, outPath):
    with open(batchPath, 'r') as fObj:
        content = fObj.readlines()
    varDict = {}
    argsList = []
    for line in content:
        line = line.strip()
        if (line==""):
            continue
        elif (line[:3]=='set'): # parse setting variable
            line = line[3:]
            line = line.replace(' ', '')
            line = line.replace("%CD%", os.path.split(batchPath)[0])
            var, value = line.split('=')
            varDict["%{}%".format(var)] = value
        elif (line[0]=='-'):
            if ('%' in line):
                for var in varDict.keys():
                    line = line.replace(var, varDict[var])
            line_split = line.split(' ')
            if (line_split[-1]=='^'):
                line_split = line_split[:-1]
            argsList.append(line_split)
    
    firstPrefixSpace = "\t\t\t"
    otherPrefixSpace = firstPrefixSpace + "\t"

    with open(outPath, 'w') as fObj:
        fObj.writelines(firstPrefixSpace + "\"args\": [\n")
        for argsSubList in argsList:
            lineWrite = str(argsSubList)[1:-1] + ',\n'
            lineWrite = lineWrite.replace("\'", "\"")
            fObj.writelines(otherPrefixSpace + lineWrite)
        fObj.writelines(otherPrefixSpace + '],')

#-----------------------------------------------------------------------------------#

if __name__ == '__main__':

    batchPathList = [
                    "./script/finetune_cifar10.bat",
                    "./script/train.bat",
                    "./script/test.bat",
                    ]

    outPathList = [ 
                    "./.vscode/args_finetune.txt",
                    "./.vscode/args_train.txt",
                    "./.vscode/args_test.txt",
                    ]

    for batchPath, outPath in zip(batchPathList, outPathList):
        ParseArgsForBatch(batchPath, outPath)
