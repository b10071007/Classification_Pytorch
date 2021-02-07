import re

#--------------------------------------------------------------------------------------------------------#

def ParseBatchToShell(src_row):
    out_row = src_row.replace("%CD%", "$PWD")
    out_row = out_row.replace("set ", "")
    out_row = out_row.replace("@REM", "#")
    out_row = out_row.replace(" ^", " \\")

    match_list = re.findall(r"%\w+%", out_row, re.MULTILINE)
    if len(match_list)>0:
        for match_string in match_list:
            var_name = match_string[1:-1]
            out_row = out_row.replace(match_string, "${" + var_name + "}")
    
    return out_row

def ParseShellToBatch(src_row):
    out_row = src_row.replace("$PWD", "%CD%")
    out_row = out_row.replace("#", "@REM", 1)
    out_row = out_row.replace(" \\", " ^")
    
    match_list = re.findall(r"\w+=", out_row)
    if len(match_list)>0:
        out_row = out_row.replace(match_list[0], "set " + match_list[0])

    if "$" in out_row:
        match_list = re.findall(r"\${\w+}", out_row, re.MULTILINE)
        if len(match_list)>0:
            for match_string in match_list:
                var_name = match_string[2:-1]
                out_row = out_row.replace(match_string, "%{}%".format(var_name))
    
    return out_row

def ConvertScript(srcPath, outPath, batch2shell=True):
    
    with open(srcPath, 'r') as fObj_src:
        src_list = fObj_src.readlines()
    out_list = []
    for src_row in src_list:
        if batch2shell:
            out_row = ParseBatchToShell(src_row)
        else:
            out_row = ParseShellToBatch(src_row)
            
        out_list.append(out_row)
    
    with open(outPath, 'w') as fObj_out:
        for out_row in out_list:
            fObj_out.writelines(out_row)

#--------------------------------------------------------------------------------------------------------#  

def main():  
    
    srcPathList = [
                    "sample/train.bat",
                    "sample/test.bat",
                    "sample/train.sh",
                    "sample/test.sh",
                    
                  ]
    outPathList = [
                    "sample/train.sh",
                    "sample/test.sh",
                    "sample/train_debug.bat",
                    "sample/test_debug.bat",
                  ]
    
    
    for srcPath, outPath in zip(srcPathList, outPathList):
        if (".bat" in srcPath) and (".sh" in outPath):
            ConvertScript(srcPath, outPath, True)
        elif (".sh" in srcPath) and (".bat" in outPath):
            ConvertScript(srcPath, outPath, False)
        else:
            raise TypeError("Error: Only support conversion between .bat and .sh")
    
#--------------------------------------------------------------------------------------------------------#  
    
if __name__ == '__main__':
    main()
    