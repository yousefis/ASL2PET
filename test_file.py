import numpy as np
import  SimpleITK as sitk
from functions.reader.data_reader import _read_data
if __name__=="__main__":
    data_path="/exports/lkeb-hpc/syousefi/Data/asl_pet/"
    '''read path of the images for train, test, and validation'''
    _rd = _read_data(data_path)
    train_data, validation_data, test_data=_rd.read_data_path()
    m_t1=.00000000000000000000001
    M_t1=1
    for d in validation_data:
        # print(d)
        t1=sitk.GetArrayFromImage(sitk.ReadImage(d['t1']))
        mt1= np.min(t1)
        Mt1= np.max(t1)
        if mt1<m_t1:
            m_t1=mt1
        if Mt1>M_t1:
            M_t1=Mt1
        print('['+str(mt1)+' , '+str(Mt1)+']')
    print('['+str(m_t1)+' , '+str(M_t1)+']')
