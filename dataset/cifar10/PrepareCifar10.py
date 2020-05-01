import os
import pickle
from PIL import Image
from datetime import datetime

from dataset import prepare

# --------------------------------------------------------------------------------------------#

def GetCurrentTime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def LoadCifar10(fName):
    with open(fName, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    batch_label = dict.get(b"batch_label")
    filenames = dict.get(b"filenames")
    data = dict.get(b"data")
    labels = dict.get(b"labels")

    return batch_label, filenames, data, labels


def ConvertCifar10(src_DatasetDir, out_DatasetDir, out_fListPath, out_testListPath):
    fNames = ["data_batch_{}".format(i) for i in range(1, 6)] + ["test_batch"]

    os.makedirs(out_DatasetDir, exist_ok=True)
    fObj = open(out_fListPath, "w")
    fObj_test = open(out_testListPath, "w")
    for i in range(len(fNames)):
        fName = fNames[i]
        print("{} Start convert {}".format(GetCurrentTime(), fName))
        batch_label, filenames, data, labels = LoadCifar10(src_DatasetDir + fName)
        data = data.reshape(-1, 3, 32, 32)
        images = data.transpose(0, 2, 3, 1).astype("uint8")

        num_images = len(filenames)
        print(" - Process [{}/{}]".format(0, num_images))
        for idx in range(num_images):
            if idx % 2000 == 1999: print(" - Process [{}/{}]".format(idx + 1, num_images))
            imgName = filenames[idx].decode()
            label = str(labels[idx])

            image = Image.fromarray(images[idx])
            image = image.convert('RGB')

            filepath = os.path.join(out_DatasetDir, label, imgName)
            os.makedirs(os.path.split(filepath)[0], exist_ok=True)
            image.save(filepath)

            if "test" in fName:
                fObj_test.write("{}/{} {}\n".format(label, imgName, label))
            else:
                fObj.write("{}/{} {}\n".format(label, imgName, label))

    fObj.close()
    fObj_test.close()

# --------------------------------------------------------------------------------------------#

def main():

    rootPath = "D:/Dataset/Classification/cifar10/"
    src_DatasetDir = "D:/Dataset/Classification/cifar-10-batches-py/"
    out_DatasetDir = rootPath + "_Images/"
    out_fListPath = rootPath + "fList.txt"
    out_testListPath = rootPath + "test.txt"

    # Convert cifar10 into images & mapping list
    # ConvertCifar10(src_DatasetDir, out_DatasetDir, out_fListPath, out_testListPath)

    # Split file list into training set & validation set
    setNameList = ['train.txt', 'val.txt']
    numList = [40000, 10000]
    # prepare.Split_set_by_class_distribution(srcFPath=out_fListPath, setNameList=setNameList, numList=numList)

    # Calculate channel means
    mapListPath = rootPath + "train.txt"
    outFilePath = rootPath + "_channel_mean_STD.txt"
    outMeanImagePath = rootPath + "_mean_image.jpg"
    new_size = [48,48]
    channel = 3

    prepare.Compute_mean(mapListPath=mapListPath, srcImagePath=out_DatasetDir, new_size=new_size, channel=channel,
                         outFilePath=outFilePath, outMeanImagePath=outMeanImagePath)
# --------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    print()
    main()
