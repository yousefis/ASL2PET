##################################################
## {}
##################################################
## {License_info}
##################################################
## Author: {Sahar Yousefi}
## Copyright: Copyright {2020}, {LUMC}
## Credits: [Sahar Yousefi]
## License: {GPL}
## Version: 1.0.0
## Mmaintainer: {Sahar Yousefi}
## Email: {s.yousefi.radi[at]lumc.nl}
## Status: {Research}
##################################################
import  SimpleITK as sitk
from functions.reader.data_reader import _read_data
if __name__=="__main__":
    data_path="/exports/lkeb-hpc/syousefi/Data/asl_pet/"
    '''read path of the images for train, test, and validation'''
    _rd = _read_data(data_path)
    train_data, validation_data, test_data=_rd.read_data_path()
    for d in train_data:
        I=sitk.ReadImage(d['t1'])
        t1=sitk.GetArrayFromImage(I)
        norm_t1=t1/100000
        sitk_I=sitk.GetImageFromArray(norm_t1)
        sitk_I.SetSpacing(I.GetSpacing())
        sitk_I.SetDirection(I.GetDirection())
        sitk_I.SetOrigin(I.GetOrigin())
        name = d['t1'].rsplit('.nii')[0]+'_norm.nii'
        sitk.WriteImage(sitk_I,name)



