import os
import numpy as np
from datetime import datetime

# setup output manager
class OutputManager():
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        filePath = os.path.split(log_file_path)[0]
        if not os.path.exists(filePath): os.makedirs(filePath)
        self.log_file = open(log_file_path, 'w')

    def output(self, content):
        print(content)
        self.log_file.write(content + '\n')

    def close(self):
        self.log_file.close()

def GetCurrentTime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(1).reshape(e_x.shape[0],1)