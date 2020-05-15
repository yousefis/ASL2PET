import numpy as np
from functions.reader.data_reader import *
from functions.reader.data_reader import _read_data
import  SimpleITK as sitk
if __name__=='__main__':
    data_path = "/exports/lkeb-hpc/syousefi/Data/asl_pet/"
    _rd = _read_data(data_path)
    train_data, validation_data, test_data=_rd.read_data_path()
    Max=[]
    for i in train_data:
        I=sitk.GetArrayFromImage(sitk.ReadImage(i["pet"]))
        M=np.max(I)
        Max.append(M)
    print(np.mean(Max))