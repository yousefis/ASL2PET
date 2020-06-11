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
import tensorflow as tf
import SimpleITK as sitk
# import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
# import matplotlib.pyplot as plt


class _read_data:
    def __init__(self,data_path_AMUC,data_path_LUMC):
        self.data_path_AMUC=data_path_AMUC
        self.data_path_LUMC=data_path_LUMC
    # ========================
    def read_image_seg_penalize_volume(self, CTs, GTVs, Torso, Penalize, img_index, ct_cube_size, gtv_cube_size):

        CT_image1 = sitk.ReadImage(''.join(CTs[int(img_index)]))
        voxel_size = CT_image1.GetSpacing()
        origin = CT_image1.GetOrigin()
        direction = CT_image1.GetDirection()

        #
        CT_image = (CT_image1)

        GTV_image = sitk.ReadImage(''.join(GTVs[int(img_index)]))
        #

        Torso_image = sitk.ReadImage(''.join(Torso[int(img_index)]))
        Penalize_image = sitk.ReadImage(''.join(Penalize[int(img_index)]))
        #

        padd_zero = 87 * 2 + 2
        crop = sitk.CropImageFilter()
        crop.SetLowerBoundaryCropSize([int(padd_zero / 2) + 1, int(padd_zero / 2) + 1,
                                       int(padd_zero / 2) + 1])
        crop.SetUpperBoundaryCropSize([int(padd_zero / 2), int(padd_zero / 2), int(padd_zero / 2)])
        CT_image = crop.Execute(CT_image)
        GTV_image = crop.Execute(GTV_image)
        Torso_image = crop.Execute(Torso_image)
        Penalize_image = crop.Execute(Penalize_image)

        # padd:
        gtv = sitk.GetArrayFromImage(GTV_image)
        one = np.where(gtv)
        c_x = int((np.min(one[0]) + np.max(one[0])) / 2)
        c_y = int((np.min(one[1]) + np.max(one[1])) / 2)
        c_z = int((np.min(one[2]) + np.max(one[2])) / 2)

        if c_x - int(ct_cube_size / 2) < 0:
            xp1 = np.abs(c_x - int(ct_cube_size / 2))
        else:
            xp1 = int(ct_cube_size) - (c_x - int(2 * c_x / ct_cube_size) * int(ct_cube_size / 2))
        if c_x + int(ct_cube_size / 2) > np.shape(gtv)[0]:
            xp2 = int(ct_cube_size / 2) - (np.abs(np.shape(gtv)[0] - c_x))
        else:
            xp2 = int(ct_cube_size) - \
                  ((np.shape(gtv)[0] - c_x) -
                   int(2 * (np.shape(gtv)[0] - c_x) / ct_cube_size) * int(ct_cube_size / 2))

        if c_y - int(ct_cube_size / 2) < 0:
            yp1 = np.abs(c_y - int(ct_cube_size / 2))
        else:
            yp1 = ct_cube_size - (
                    (c_y - int(ct_cube_size / 2)) - (
                        int((c_y - int(ct_cube_size / 2)) / int(ct_cube_size)) * ct_cube_size))
        if c_y + int(ct_cube_size / 2) > np.shape(gtv)[1]:
            yp2 = np.abs((np.shape(gtv)[1] - c_y) - int(ct_cube_size / 2))
        else:
            yp2 = ct_cube_size - ((np.shape(gtv)[1] - (c_y + int(ct_cube_size / 2))) -
                                  np.floor(
                                      (np.shape(gtv)[1] - (c_y + int(ct_cube_size / 2))) / ct_cube_size) * ct_cube_size)

        if c_z - int(ct_cube_size / 2) < 0:
            zp1 = np.abs(c_z - int(ct_cube_size / 2))
        else:
            zp1 = ct_cube_size - (
                    (c_z - int(ct_cube_size / 2)) - (
                        int((c_z - int(ct_cube_size / 2)) / int(ct_cube_size)) * ct_cube_size))
        if c_z + int(ct_cube_size / 2) > np.shape(gtv)[2]:
            zp2 = np.abs((np.shape(gtv)[2] - c_z) - int(ct_cube_size / 2))
        else:
            zp2 = ct_cube_size - ((np.shape(gtv)[2] - (c_z + int(ct_cube_size / 2))) -
                                  np.floor(
                                      (np.shape(gtv)[2] - (c_z + int(ct_cube_size / 2))) / ct_cube_size) * ct_cube_size)

        CT_image = self.image_padding(img=CT_image,
                                      padLowerBound=[int(yp1 + 1), int(zp1 + 1), int(xp1 + 1)],
                                      padUpperBound=[int(yp2), int(zp2), int(xp2)],
                                      constant=-1024)
        GTV_image = self.image_padding(img=GTV_image,
                                       padLowerBound=[int(yp1 + 1), int(zp1 + 1), int(xp1 + 1)],
                                       padUpperBound=[int(yp2), int(zp2), int(xp2)],
                                       constant=0)
        Torso_image = self.image_padding(img=Torso_image,
                                         padLowerBound=[int(yp1 + 1), int(zp1 + 1), int(xp1 + 1)],
                                         padUpperBound=[int(yp2), int(zp2), int(xp2)],
                                         constant=0)
        Penalize_image = self.image_padding(img=Penalize_image,
                                            padLowerBound=[int(yp1 + 1), int(zp1 + 1), int(xp1 + 1)],
                                            padUpperBound=[int(yp2), int(zp2), int(xp2)],
                                            constant=0)
        # ----------------------------------------
        ct = sitk.GetArrayFromImage(CT_image)
        gtv1 = sitk.GetArrayFromImage(GTV_image)
        # Torso_image = sitk.GetArrayFromImage(Torso_image)
        c = 0
        gap = ct_cube_size - gtv_cube_size

        for _z in (
                range(int(ct_cube_size / 2) + 1, ct.shape[0] - int(ct_cube_size / 2) + 7,
                      int(ct_cube_size) - int(gap) + 1)):
            for _x in (range(int(ct_cube_size / 2) + 1, ct.shape[1] - int(ct_cube_size / 2) + 7,
                             int(ct_cube_size) - int(gap) + 1)):
                for _y in (range(int(ct_cube_size / 2) + 1, ct.shape[2] - int(ct_cube_size / 2) + 7,
                                 int(ct_cube_size) - int(gap) + 1)):
                    gtv = gtv1[_z - int(gtv_cube_size / 2) - 1:_z + int(gtv_cube_size / 2),
                          _x - int(gtv_cube_size / 2) - 1:_x + int(gtv_cube_size / 2),
                          _y - int(gtv_cube_size / 2) - 1:_y + int(gtv_cube_size / 2)]
                    if len(np.where(gtv[:, :, :] != 0)[0]):
                        print(len(np.where(gtv[:, :, :] != 0)[0]))
                        c = c + 1
        if c != 1:
            print('hhhhhhhhhhhhhhhhhhhhhhh')
        # ----------------------------------------

        CT_image = sitk.GetArrayFromImage(CT_image)
        GTV_image = sitk.GetArrayFromImage(GTV_image)
        Torso_image = sitk.GetArrayFromImage(Torso_image)
        Penalize_image = sitk.GetArrayFromImage(Penalize_image)

        return CT_image, GTV_image, Torso_image, Penalize_image, GTV_image.shape[0], voxel_size, origin, direction

    # ========================
    def read_data_path(self,average_no=52):  # join(self.resampled_path, f)
        print('Reading images from hard drive!')
        # read LUMC data
        data_dir_lumc = [join(self.data_path_LUMC, f)
                         for f in listdir(self.data_path_LUMC)
                         if (not (isfile(join(self.data_path_LUMC, f))) and
                             not (os.path.isfile(join(self.data_path_LUMC, f + '/delete.txt'))))]
        data_dir_lumc.sort()
        triple_data_lumc = []
        tags = ['checkerboard.nii', 'motor.nii', 'rest.nii', 'tomenjerry.nii']
        for pp in triple_data_lumc:


        #read AMUC data
        data_dir_amuc = [join(self.data_path_AMUC, f)
                         for f in listdir(self.data_path_AMUC)
                         if (not (isfile(join(self.data_path_AMUC, f))) and
                        not (os.path.isfile(join(self.data_path_AMUC, f + '/delete.txt'))))]
        data_dir_amuc.sort()
        triple_data_amuc=[]
        tags= ['_W1_HN1','_W1_HN2','_W1_HY','_W6_HN','_W6_HY']
        for pp in data_dir_amuc:
            asl_dir=pp+'/ASL_'+str(average_no)+'/'
            asls = [join(asl_dir, f)
                 for f in listdir(asl_dir)
                 if ( (isfile(join(asl_dir, f))) )   ]
            asls.sort()
            pet_dir = pp + '/PET/'
            pets = [join(pet_dir, f)
                    for f in listdir(pet_dir)
                    if ( (isfile(join(pet_dir, f))))]
            pets.sort()
            t1_dir = pp + '/T1/T1.nii'
            mask = pp + '/T1/T1_brain_mask.nii'

            for i in range(5):
                asl=asl_dir+'ASL'+tags[i]+'.nii'
                pet=pet_dir+'PET'+tags[i]+'.nii'
                if asl in asls and pet in pets:
                    triple_atom_amuc= {'t1':t1_dir,'asl':asl,'pet':pet,'mask':mask}
                    triple_data_amuc.append(triple_atom_amuc)
                elif asl in asls and not pet in pets:
                    continue #for now we are not going to use these cases
                    triple_atom_amuc = {'t1': t1_dir, 'asl': asl, 'pet': None}
                    triple_data_amuc.append(triple_atom_amuc)
                elif not asl in asls and pet in pets:
                    continue #for now we are not going to use these cases
                    triple_atom_amuc = {'t1': t1_dir, 'asl': None, 'pet': pet}
                    triple_data_amuc.append(triple_atom_amuc)

        test_data_amuc= triple_data_amuc[:10]
        validation_data_amuc = triple_data_amuc[10:15]
        trian_data_amuc= triple_data_amuc[15:]






        return trian_data_amuc, validation_data_amuc, test_data_amuc

    # ========================
    def read_image_path3(self, image_path):  # for padding_images
        CTs = []
        GTVs = []
        Torsos = []
        resampletag = '_re113'
        img_name = 'CT' + resampletag + 'z.mha'
        label_name = 'GTV_CT' + resampletag + '.mha'
        label_name2 = 'GTV' + resampletag + '.mha'
        torso_tag = 'CT_Torso' + resampletag + '.mha'

        startwith_4DCT = '4DCT_'
        startwith_GTV = 'GTV_'

        img_name = img_name
        label_name = label_name
        torso_tag = torso_tag

        data_dir = [join(image_path, f) for f in listdir(image_path) if ~isfile(join(image_path, f))]
        for pd in data_dir:
            date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                    ~isfile(join(image_path, pd, dt))]
            for dt in date:
                # read CT and GTV images
                CT_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                           f.startswith(img_name)]
                GTV_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                            f.endswith(label_name) or f.endswith(label_name2)]
                # print GTV_path
                Torso_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                              f.endswith(torso_tag)]

                # print('%s\n%s\n%s' % (
                # CT_path[len(GTV_path) - 1], GTV_path[len(GTV_path) - 1], Torso_path[len(GTV_path) - 1]))

                CT4D_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                             f.startswith(startwith_4DCT) & f.endswith('%' + self.resample_tag + 'z.mha')]

                CT_path = CT_path + CT4D_path  # here sahar
                for i in range(len(CT4D_path)):
                    # print CT4D_path[i]
                    name_gtv4d = 'GTV_' + CT4D_path[i].split('/')[10].split('z.')[0] + '.mha'
                    # print('name:%s'%(name_gtv4d))
                    GTV_path.append((str(join(image_path, pd, dt, name_gtv4d))))  # here sahar
                    Torso_gtv4d = CT4D_path[i].split('/')[10].split(self.resample_tag + 'z.')[
                                      0] + '_Torso' + self.resample_tag + '.mha'
                    # print('***********'+Torso_gtv4d)
                    Torso_path.append((str(join(image_path, pd, dt, Torso_gtv4d))))
                # print (';;;;'+CT4D_path[i])

                # print('%s\n%s\n%s'%(CT_path[len(GTV_path)-1],GTV_path[len(GTV_path)-1],Torso_path[len(GTV_path)-1]))

                CTs += (CT_path)
                GTVs += (GTV_path)
                Torsos += (Torso_path)
        return CTs, GTVs, Torsos

    # ==========================
    def read_image_path2(self, image_path):  # for mask_image_by_torso
        CTs = []
        GTVs = []
        Torsos = []
        resampletag = '_re113'
        img_name = 'CT' + resampletag + 'z' + '.mha'
        label_name = 'GTV_CT' + resampletag +  '.mha'
        label_name2 = 'GTV' + resampletag +  '.mha'
        torso_tag = 'CT_Torso' + resampletag  + '.mha'

        startwith_4DCT = '4DCT_'
        startwith_GTV = 'GTV_'

        img_name = img_name
        label_name = label_name
        torso_tag = torso_tag

        data_dir = [join(image_path, f) for f in listdir(image_path) if ~isfile(join(image_path, f))]
        for pd in data_dir:
            date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                    ~isfile(join(image_path, pd, dt))]
            for dt in date:
                # read CT and GTV images
                CT_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                           f.startswith(img_name)]
                GTV_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                            f.endswith(label_name) or f.endswith(label_name2)]
                # print GTV_path
                Torso_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                              f.endswith(torso_tag)]

                # print('%s\n%s\n%s' % (
                # CT_path[len(GTV_path) - 1], GTV_path[len(GTV_path) - 1], Torso_path[len(GTV_path) - 1]))

                CT4D_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                             f.startswith(startwith_4DCT) & f.endswith(
                                 '%' + self.resample_tag + 'z_pad' + str(self.patchsize) + '.mha')]

                CT_path = CT_path + CT4D_path  # here sahar
                for i in range(len(CT4D_path)):
                    # print CT4D_path[i]
                    name_gtv4d = 'GTV_' + CT4D_path[i].split('/')[10].split('z_pad' + str(self.patchsize) + '.')[
                        0] +  '.mha'
                    # print('name:%s'%(name_gtv4d))
                    GTV_path.append((str(join(image_path, pd, dt, name_gtv4d))))  # here sahar
                    Torso_gtv4d = CT4D_path[i].split('/')[10].split('_re113' + '.')[
                                      0] + '_Torso' + self.resample_tag + '.mha'
                    # print('***********'+Torso_gtv4d)
                    Torso_path.append((str(join(image_path, pd, dt, Torso_gtv4d))))
                # print (';;;;'+CT4D_path[i])

                # print('%s\n%s\n%s'%(CT_path[len(GTV_path)-1],GTV_path[len(GTV_path)-1],Torso_path[len(GTV_path)-1]))

                CTs += (CT_path)
                GTVs += (GTV_path)
                Torsos += (Torso_path)
        return CTs, GTVs, Torsos

    # ========================
    def read_volume(self, path):
        ct = sitk.ReadImage(path)
        voxel_size = ct.GetSpacing()
        origin = ct.GetOrigin()
        direction = ct.GetDirection()
        ct = sitk.GetArrayFromImage(ct)
        return ct, voxel_size, origin, direction

    # ========================
    def read_image_path(self, image_path):
        CTs = []
        GTVs = []
        Torsos = []
        data_dir = [join(image_path, f) for f in listdir(image_path) if ~isfile(join(image_path, f))]
        if self.data == 1:
            for pd in data_dir:
                date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                        ~isfile(join(image_path, pd, dt))]
                for dt in date:
                    CT_path = [(join(image_path, pd, dt, self.prostate_ext_img, f)) for f in
                               listdir(join(image_path, pd, dt, self.prostate_ext_img)) if
                               f.endswith(self.img_name)]
                    GTV_path = [join(image_path, pd, dt, self.prostate_ext_gt, f) for f in
                                listdir(join(image_path, pd, dt, self.prostate_ext_gt)) if
                                f.endswith(self.label_name)]

                    CTs.append(CT_path)
                    GTVs.append(GTV_path)

            return CTs, GTVs, Torsos
        elif self.data == 2:
            for pd in data_dir:
                date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                        ~isfile(join(image_path, pd, dt))]

                for dt in date:

                    # read CT and GTV images
                    CT_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                               f.startswith(self.img_name)]
                    GTV_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                                f.endswith(self.label_name)]
                    Torso_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                                  f.endswith(self.torso_tag)]

                    # print('%s\n%s\n%s' % (
                    # CT_path[len(GTV_path) - 1], GTV_path[len(GTV_path) - 1], Torso_path[len(GTV_path) - 1]))

                    CT4D_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                                 f.startswith(self.startwith_4DCT) & f.endswith('_padded.mha')]
                    CT_path = CT_path + CT4D_path  # here sahar
                    for i in range(len(CT4D_path)):
                        name_gtv4d = 'GTV_4DCT_' + CT4D_path[i].split('/')[10].split('.')[0].split('_')[
                            1] + '_padded.mha'
                        GTV_path.append((str(join(image_path, pd, dt, name_gtv4d))))  # here sahar
                        Torso_gtv4d = CT4D_path[i].split('/')[10].split('.')[0] + '_Torso.mha'
                        Torso_path.append((str(join(image_path, pd, dt, Torso_gtv4d))))

                        # print('%s\n%s\n%s'%(CT_path[len(GTV_path)-1],GTV_path[len(GTV_path)-1],Torso_path[len(GTV_path)-1]))

                    CTs += (CT_path)
                    GTVs += (GTV_path)
                    Torsos += (Torso_path)

            return CTs, GTVs, Torsos

    def read_image_path2(self):
        '''read the name of all the images and annotaed image'''
        print('Looking for the normal constants, please wait...')
        train_CTs, train_GTVs, train_Torso = self.read_image_path(self.train_image_path)
        [min1, max1] = self.return_normal_const(train_CTs)

        validation_CTs, validation_GTVs, validation_Torso = self.read_image_path(self.validation_image_path)
        [min2, max2] = self.return_normal_const(validation_CTs)

        test_CTs, test_GTVs, test_Torso = self.read_image_path(self.test_image_path)
        [min3, max3] = self.return_normal_const(test_CTs)

        min_normal = np.min([min1, min2, min3])
        max_normal = np.max([max1, max2, max3])
        # [depth,width,height]=self.return_depth_width_height( CTs)
        return train_CTs, train_GTVs, train_Torso, validation_CTs, validation_GTVs, validation_Torso, \
               test_CTs, test_GTVs, test_Torso, min_normal, max_normal

    def read_image_path(self):
        '''read the name of all the images and annotaed image'''
        train_CTs, train_GTVs, train_Torso = self.read_image_path(self.train_image_path)

        validation_CTs, validation_GTVs, validation_Torso = self.read_image_path(self.validation_image_path)

        test_CTs, test_GTVs, test_Torso = self.read_image_path(self.test_image_path)

        [depth, width, height] = self.return_depth_width_height(train_CTs)
        return train_CTs, train_GTVs, train_Torso, validation_CTs, validation_GTVs, validation_Torso, \
               test_CTs, test_GTVs, test_Torso, depth, width, height

    def return_depth_width_height(self, CTs):
        CT_image = sitk.ReadImage(''.join(CTs[int(0)]))
        CT_image = sitk.GetArrayFromImage(CT_image)
        return CT_image.shape[0], CT_image.shape[1], CT_image.shape[2]

    def return_normal_const(self, CTs):
        min_normal = 1E+10
        max_normal = -min_normal

        for i in range(len(CTs)):
            CT_image = sitk.ReadImage(''.join(CTs[int(i)]))
            CT_image = sitk.GetArrayFromImage(CT_image)
            max_tmp = np.max(CT_image)
            if max_tmp > max_normal:
                max_normal = max_tmp
            min_tmp = np.min(CT_image)
            if min_tmp < min_normal:
                min_normal = min_tmp
        return min_normal, max_normal

    # =================================================================
    def image_padding(self, img, padLowerBound, padUpperBound, constant):
        filt = sitk.ConstantPadImageFilter()
        padded_img = filt.Execute(img,
                                  padLowerBound,
                                  padUpperBound,
                                  constant)
        return padded_img

    # =================================================================

    def read_image_seg_volume(self, CTs, GTVs, Torso, img_index, ct_cube_size, gtv_cube_size):

        CT_image1 = sitk.ReadImage(''.join(CTs[int(img_index)]))
        voxel_size = CT_image1.GetSpacing()
        origin = CT_image1.GetOrigin()
        direction = CT_image1.GetDirection()

        #
        CT_image = (CT_image1)  # /CT_image.mean()

        GTV_image = sitk.ReadImage(''.join(GTVs[int(img_index)]))
        #

        Torso_image = sitk.ReadImage(''.join(Torso[int(img_index)]))
        #

        padd_zero = 87 * 2 + 2
        crop = sitk.CropImageFilter()
        crop.SetLowerBoundaryCropSize([int(padd_zero / 2) + 1, int(padd_zero / 2) + 1,
                                       int(padd_zero / 2) + 1])
        crop.SetUpperBoundaryCropSize([int(padd_zero / 2), int(padd_zero / 2), int(padd_zero / 2)])
        CT_image = crop.Execute(CT_image)
        GTV_image = crop.Execute(GTV_image)
        Torso_image = crop.Execute(Torso_image)

        # padd:
        gtv = sitk.GetArrayFromImage(GTV_image)
        one = np.where(gtv)
        c_x = int((np.min(one[0]) + np.max(one[0])) / 2)
        c_y = int((np.min(one[1]) + np.max(one[1])) / 2)
        c_z = int((np.min(one[2]) + np.max(one[2])) / 2)

        if c_x - int(ct_cube_size / 2) < 0:
            xp1 = np.abs(c_x - int(ct_cube_size / 2))
        else:
            xp1 = int(ct_cube_size) - (c_x - int(2 * c_x / ct_cube_size) * int(ct_cube_size / 2))
        if c_x + int(ct_cube_size / 2) > np.shape(gtv)[0]:
            xp2 = int(ct_cube_size / 2) - (np.abs(np.shape(gtv)[0] - c_x))
        else:
            xp2 = int(ct_cube_size) - \
                  ((np.shape(gtv)[0] - c_x) -
                   int(2 * (np.shape(gtv)[0] - c_x) / ct_cube_size) * int(ct_cube_size / 2))

        if c_y - int(ct_cube_size / 2) < 0:
            yp1 = np.abs(c_y - int(ct_cube_size / 2))
        else:
            yp1 = ct_cube_size - (
            (c_y - int(ct_cube_size / 2)) - (int((c_y - int(ct_cube_size / 2)) / int(ct_cube_size)) * ct_cube_size))
        if c_y + int(ct_cube_size / 2) > np.shape(gtv)[1]:
            yp2 = np.abs((np.shape(gtv)[1] - c_y) - int(ct_cube_size / 2))
        else:
            yp2 = ct_cube_size - ((np.shape(gtv)[1] - (c_y + int(ct_cube_size / 2))) -
                                  np.floor(
                                      (np.shape(gtv)[1] - (c_y + int(ct_cube_size / 2))) / ct_cube_size) * ct_cube_size)

        if c_z - int(ct_cube_size / 2) < 0:
            zp1 = np.abs(c_z - int(ct_cube_size / 2))
        else:
            zp1 = ct_cube_size - (
            (c_z - int(ct_cube_size / 2)) - (int((c_z - int(ct_cube_size / 2)) / int(ct_cube_size)) * ct_cube_size))
        if c_z + int(ct_cube_size / 2) > np.shape(gtv)[2]:
            zp2 = np.abs((np.shape(gtv)[2] - c_z) - int(ct_cube_size / 2))
        else:
            zp2 = ct_cube_size - ((np.shape(gtv)[2] - (c_z + int(ct_cube_size / 2))) -
                                  np.floor(
                                      (np.shape(gtv)[2] - (c_z + int(ct_cube_size / 2))) / ct_cube_size) * ct_cube_size)



        CT_image = self.image_padding(img=CT_image,
                                      padLowerBound=[int(yp1 + 1), int(zp1 + 1), int(xp1 + 1)],
                                      padUpperBound=[int(yp2), int(zp2), int(xp2)],
                                      constant=-1024)
        GTV_image = self.image_padding(img=GTV_image,
                                       padLowerBound=[int(yp1 + 1), int(zp1 + 1), int(xp1 + 1)],
                                       padUpperBound=[int(yp2), int(zp2), int(xp2)],
                                       constant=0)
        Torso_image = self.image_padding(img=Torso_image,
                                         padLowerBound=[int(yp1 + 1), int(zp1 + 1), int(xp1 + 1)],
                                         padUpperBound=[int(yp2), int(zp2), int(xp2)],
                                         constant=0)

        # ----------------------------------------
        ct = sitk.GetArrayFromImage(CT_image)
        gtv1 = sitk.GetArrayFromImage(GTV_image)
        # Torso_image = sitk.GetArrayFromImage(Torso_image)
        c = 0
        gap = ct_cube_size - gtv_cube_size

        for _z in (
                range(int(ct_cube_size / 2) + 1, ct.shape[0] - int(ct_cube_size / 2) + 7,
                      int(ct_cube_size) - int(gap) + 1)):
            for _x in (range(int(ct_cube_size / 2) + 1, ct.shape[1] - int(ct_cube_size / 2) + 7,
                             int(ct_cube_size) - int(gap) + 1)):
                for _y in (range(int(ct_cube_size / 2) + 1, ct.shape[2] - int(ct_cube_size / 2) + 7,
                                 int(ct_cube_size) - int(gap) + 1)):
                    gtv = gtv1[_z - int(gtv_cube_size / 2) - 1:_z + int(gtv_cube_size / 2),
                          _x - int(gtv_cube_size / 2) - 1:_x + int(gtv_cube_size / 2),
                          _y - int(gtv_cube_size / 2) - 1:_y + int(gtv_cube_size / 2)]
                    if len(np.where(gtv[:, :, :] != 0)[0]):
                        print(len(np.where(gtv[:, :, :] != 0)[0]))
                        c = c + 1

        CT_image = sitk.GetArrayFromImage(CT_image)
        GTV_image = sitk.GetArrayFromImage(GTV_image)
        Torso_image = sitk.GetArrayFromImage(Torso_image)

        return CT_image, GTV_image, Torso_image, GTV_image.shape[0], voxel_size, origin, direction

    # =================================================================
    def read_image(self, CT_image, GTV_image, img_height, img_padded_size, seg_size, depth):
        img = CT_image[depth, 0:img_height - 1, 0:img_height - 1]
        img1 = np.zeros((1, img_padded_size, img_padded_size))
        fill_val = img[0][0]
        img1[0][:][:] = np.lib.pad(img, (
            int((img_padded_size - img_height) / 2 + 1), int((img_padded_size - img_height) / 2 + 1)),
                                   "constant", constant_values=(fill_val, fill_val))
        img = img1[..., np.newaxis]
        seg1 = (GTV_image[depth, int(img_height / 2) - int(seg_size / 2) - 1:int(img_height / 2) + int(seg_size / 2),
                int(img_height / 2) - int(seg_size / 2) - 1:int(img_height / 2) + int(seg_size / 2)])
        seg = np.eye(2)[seg1]
        seg = seg[np.newaxis]
        return img, seg

    # =================================================================
    def check(self, GTVs, width_patch, height_patch, depth_patch):
        no_of_images = len(GTVs)
        for ii in range(no_of_images):
            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            if (max(depth_patch[ii]) > len(GTV_image)):
                print('error')


        # =================================================================

    def shuffle_lists(self, rand_width1, rand_height1, rand_depth1):
        index_shuf = list(range(len(rand_width1)))
        shuffle(index_shuf)
        rand_width11 = np.hstack([rand_width1[sn]]
                                 for sn in index_shuf)
        rand_depth11 = np.hstack([rand_depth1[sn]]
                                 for sn in index_shuf)
        rand_height11 = np.hstack([rand_height1[sn]]
                                  for sn in index_shuf)
        return rand_width11, rand_height11, rand_depth11



