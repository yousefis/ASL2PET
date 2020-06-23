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
import numpy as np
import SimpleITK as sitk
import collections
from random import shuffle
from functions.settings import settings as settings
import random
from scipy import ndimage
import random

class image_class:
    def __init__(self,scans,inp_size,out_size,
                 bunch_of_images_no=5,is_training=1):
        self.bunch_of_images_no = bunch_of_images_no
        self.node = collections.namedtuple('node', 'name_t1 name_asl name_pet t1 asl pet voxel_size origin direction')
        self.collection=[]
        self.is_training=is_training
        self.scans=scans
        self.random_images=list(range(0,len(self.scans)))
        self.random_images1 = list(range(0, 58))
        self.random_images2 = list(range(58, len(self.scans)))
        self.counter_save=0
        self.static_counter_vl=0
        self.seed = 100
        self.inp_size=inp_size
        self.out_size=out_size




    # --------------------------------------------------------------------------------------------------------
    def image_padding(self,img, padLowerBound, padUpperBound, constant):
        filt = sitk.ConstantPadImageFilter()
        padded_img = filt.Execute(img,
                                  padLowerBound,
                                  padUpperBound,
                                  constant)
        return padded_img

    def image_crop(self,img, padLowerBound, padUpperBound):
        crop_filt = sitk.CropImageFilter()
        cropped_img = crop_filt.Execute(img, padLowerBound, padUpperBound)
        return cropped_img






    # --------------------------------------------------------------------------------------------------------
    def Flip(self,CT_image, GTV_image, Torso_image):
        TF1=False
        TF2=bool(random.getrandbits(1))
        TF3=bool(random.getrandbits(1))

        CT_image = sitk.Flip(CT_image, [TF1, TF2, TF3])
        GTV_image = sitk.Flip(GTV_image, [TF1, TF2, TF3])
        Torso_image = sitk.Flip(Torso_image, [TF1, TF2, TF3])
        return CT_image, GTV_image, Torso_image


  
    # --------------------------------------------------------------------------------------------------------
    #read information of each image
    def read_image(self,s):
        T1= sitk.ReadImage(''.join(s['t1']))
        voxel_size = T1.GetSpacing()
        origin = T1.GetOrigin()
        direction = T1.GetDirection()
        # mask = sitk.GetArrayFromImage(sitk.ReadImage(''.join(s['mask'])))

        # t1 = np.multiply(sitk.GetArrayFromImage(T1),mask)/1000
        # asl = np.multiply(sitk.GetArrayFromImage(sitk.ReadImage(''.join(s['asl']))),mask)
        # pet = np.multiply(sitk.GetArrayFromImage(sitk.ReadImage(''.join(s['pet']))),mask)

        t1 = sitk.GetArrayFromImage(T1)
        t1 = t1 / np.max(t1)
        asl = sitk.GetArrayFromImage(sitk.ReadImage(''.join(s['asl'])))
        # asl = asl / np.max(asl)
        if s['pet']==None:
            pet=None
        else:
            pet = sitk.GetArrayFromImage(sitk.ReadImage(''.join(s['pet'])))
            pet[np.where(pet < 0)] = 0
            pet = pet[1:-1]

        
        t1[np.where(t1<0)]=0
        asl[np.where(asl < 0)] = 0

        
        t1 = t1[1:-1]
        asl = asl[1:-1]


        n = self.node(name_t1=s['t1'],name_asl=s['asl'],name_pet=s['pet'], t1=t1, asl=asl, pet=pet,
                      voxel_size=voxel_size, origin=origin, direction=direction)
        return n

    def return_normal_image(self,CT_image,max_range,min_range,min_normal,max_normal):
        return (max_range - min_range) * (
        (CT_image - min_normal) / (max_normal - min_normal)) + min_range
    # --------------------------------------------------------------------------------------------------------
    def random_gen(self,low, high):
        while True:
            yield random.randrange(low, high)
    # --------------------------------------------------------------------------------------------------------


    # --------------------------------------------------------------------------------------------------------
    def read_bunch_of_images(self):  # for training
        if settings.tr_isread==False:
            return
        settings.read_patche_mutex_tr.acquire()
        self.collection.clear()
        self.seed += 1
        np.random.seed(self.seed)

        if len(self.random_images1) < self.bunch_of_images_no:  # if there are no image choices for selection
            self.random_images1 = list(range(0,58))
            self.random_images2 = list(range(58, len(self.scans)))
            settings.epochs_no+=1

        # select some distinct images for extracting patches!
        r1=np.random.randint(0, len(self.random_images1), int(self.bunch_of_images_no/2)+1)
        r2=np.random.randint(0, len(self.random_images2), int(self.bunch_of_images_no / 2))
        rand_image_no= np.hstack(([self.random_images1[x] for x in r1],
                                  [self.random_images2[x] for x in r2]))
        # rand_image_no=

        self.random_images1 = [x for x in range(len(self.random_images1)) if
                              x not in rand_image_no]  # remove selected images from the choice list
        self.random_images2 = [x for x in range(len(self.random_images2)) if
                               x not in rand_image_no]  # remove selected images from the choice list
        print(rand_image_no)


        aa=0
        bb=0
        for img_index in range(len(rand_image_no)):


            imm = self.read_image(self.scans[rand_image_no[img_index]])
            if len(imm) == 0:

                continue
            if imm['pet'] is None:
                aa=aa+1
            else:
                bb=bb+1
            print(aa,bb)

            self.collection.append(imm)
            print('train image no read so far: %s'%len(self.collection))

        settings.tr_isread=False
        settings.read_patche_mutex_tr.release()




    # --------------------------------------------------------------------------------------------------------
    # read images and transfer those to RAM
    def read_bunch_of_images_vl(self, total_sample_no):  # for validation
        if len(settings.bunch_pet_slices_vl) > total_sample_no:
            return
        if settings.vl_isread == False:
            return
        settings.read_patche_mutex_vl.acquire()
        self.collection.clear()
        self.seed += 1
        np.random.seed(self.seed)
        self.random_images = list(range(0, len(self.scans)))
        for s in self.scans:
            imm = self.read_image(s)
            if len(imm) == 0:
                continue

            self.collection.append(imm)
        print('Reading the validation set was finished!')
        settings.vl_isread = False
        settings.read_patche_mutex_vl.release()

    # --------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------------
        # shuffling the patches
    def shuffle_lists(self, CT_image_patchs, pet_slicess,t1_slicess):
            index_shuf = list(range(len(pet_slicess)))
            shuffle(index_shuf)
            CT_image_patchs1 = np.vstack([CT_image_patchs[sn]]
                                         for sn in index_shuf)
            pet_slicess1 = np.vstack([pet_slicess[sn]]
                                    for sn in index_shuf)

            t1_slicess1 = np.vstack([t1_slicess[sn]]
                                    for sn in index_shuf)#np.vstack([t1_slicess[sn]] for sn in index_shuf)
            return CT_image_patchs1, pet_slicess1,t1_slicess1
    #--------------------------------------------------------------------------------------------------------

    def read_patche_online_from_image_bunch_vl(self, sample_no_per_bunch,  img_no):
        if settings.vl_isread == True:
            return

        if len(self.collection) <img_no:
            return
        self.seed += 1
        np.random.seed(self.seed)
        settings.read_patche_mutex_vl.acquire()
        # print('start reading:%d' % len(self.collection))
        # patch_no_per_image = int(sample_no_per_bunch / len(self.collection))
        # if patch_no_per_image==0:
        #     patch_no_per_image=1
        ASL = []
        PET = []
        T1 = []
        for ii in range(len(self.collection)):
            t1 = self.collection[ii].t1
            asl = self.collection[ii].asl
            pet = self.collection[ii].pet
            voxel_size = self.collection[ii].voxel_size
            origin = self.collection[ii].origin
            direction = self.collection[ii].direction
            name_t1 = self.collection[ii].name_t1
            name_asl = self.collection[ii].name_asl
            name_pet = self.collection[ii].name_pet

            print(self.collection[ii].name_t1)

            '''random numbers for selecting random samples'''
            random_slices_indx = list(range(0,15))
            size_img = np.shape(asl)[1]
            ASL1 = [np.stack(asl[random_slices_indx[sn], int(size_img/2)-int(self.inp_size/2)-1:int(size_img/2)+int(self.inp_size/2),int(size_img/2)-int(self.inp_size/2)-1:int(size_img/2)+int(self.inp_size/2)])
                    for sn in range(len(random_slices_indx))]

            PET1 = [np.stack(pet[random_slices_indx[sn],int(size_img/2)-int(self.out_size/2)-1:int(size_img/2)+int(self.out_size/2),int(size_img/2)-int(self.out_size/2)-1:int(size_img/2)+int(self.out_size/2)])
                    for sn in range(len(random_slices_indx))]


            T11 = [np.stack(t1[random_slices_indx[sn], int(size_img/2)-int(self.inp_size/2)-1:int(size_img/2)+int(self.inp_size/2),int(size_img/2)-int(self.inp_size/2)-1:int(size_img/2)+int(self.inp_size/2)])
                   for sn in range(len(random_slices_indx))]

            if len(ASL) == 0:
                ASL = np.copy(ASL1)
                PET = np.copy(PET1)
                T1 = np.copy(T11)

            else:
                ASL = np.vstack((ASL, ASL1))
                PET = np.vstack((PET, PET1))
                T1 = np.vstack((T1, T11))

        ASL1, PET1, T11 = self.shuffle_lists(ASL, PET, T1)

        if self.is_training == 1:

            settings.bunch_asl_slices2 = np.copy(ASL1)
            settings.bunch_pet_slices2 = np.copy(PET1)
            settings.bunch_t1_slices2 = np.copy(T11)

        else:

            if len(settings.bunch_pet_slices_vl2) == 0:
                settings.bunch_asl_slices_vl2 = np.copy(ASL1)
                settings.bunch_pet_slices_vl2 = np.copy(PET1)
                settings.bunch_t1_slices_vl2 = np.copy(T11)
            else:
                settings.bunch_asl_slices_vl2 = np.vstack((settings.bunch_asl_slices_vl2, ASL1))
                settings.bunch_pet_slices_vl2 = np.vstack((settings.bunch_pet_slices_vl2, PET1))
                settings.bunch_t1_slices_vl2 = np.vstack((settings.bunch_t1_slices_vl2, T11))


        settings.vl_isread=True
        if len(settings.bunch_asl_slices_vl2) != len(
                settings.bunch_pet_slices_vl2) or len(settings.bunch_asl_slices_vl2) != len(
                settings.bunch_t1_slices_vl2):  # or len(settings.bunch_t1_slices_vl2)!=len(settings.bunch_pet_slices_vl2 ):
            print('smth wrong')
        settings.read_patche_mutex_vl.release()



    #--------------------------------------------------------------------------------------------------------
    #read patches from the images which are in the RAM
    def read_patche_online_from_image_bunch(self, sample_no_per_bunch,img_no):

        if len(self.collection)<img_no:
            return
        if settings.tr_isread == True:
            return
        if len(settings.bunch_t1_slices)>200:
            return
        self.seed += 1
        np.random.seed(self.seed)
        settings.read_patche_mutex_tr.acquire()
        # print('start reading:%d'%len(self.collection))
        patch_no_per_image=int(sample_no_per_bunch/len(self.collection) ) #patch means slice here!
        while patch_no_per_image*len(self.collection)<=sample_no_per_bunch:
            patch_no_per_image+=1
        ASL=[]
        PET=[]
        T1=[]
        aa=0
        bb=0
        if len(self.collection):
            for i in range(len(self.collection)):
                if self.collection[i].pet is None:
                    aa=aa+1
                else:
                    bb=bb+1

            print(aa,bb)


        for ii in range(len(self.collection) ):
            t1 = self.collection[ii].t1
            asl = self.collection[ii].asl
            pet = self.collection[ii].pet
            # voxel_size= self.collection[ii].voxel_size
            # origin= self.collection[ii].origin
            # direction = self.collection[ii].direction
            # name_t1 = self.collection[ii].name_t1
            # name_asl = self.collection[ii].name_asl
            # name_pet = self.collection[ii].name_pet


            '''random numbers for selecting random samples'''
            random_slices_indx=np.random.randint(1,np.shape(t1)[0] ,
                                           size=int(patch_no_per_image ))




            size_img= np.shape(asl)[1]
            ASL1 = [np.stack( asl[random_slices_indx[sn], int(size_img/2)-int(self.inp_size/2)-1:int(size_img/2)+int(self.inp_size/2),int(size_img/2)-int(self.inp_size/2)-1:int(size_img/2)+int(self.inp_size/2)])
                 for sn in range(len(random_slices_indx))]
            if pet is not None:
                PET1 = [np.stack( pet[random_slices_indx[sn], int(size_img/2)-int(self.out_size/2)-1:int(size_img/2)+int(self.out_size/2),int(size_img/2)-int(self.out_size/2)-1:int(size_img/2)+int(self.out_size/2)])
                 for sn in range(len(random_slices_indx))]
            else:
                PET1 = np.reshape(len(ASL1) * self.out_size*self.out_size*[None],(len(ASL1),self.out_size,self.out_size))
            T11 = [np.stack(t1[random_slices_indx[sn],  int(size_img/2)-int(self.inp_size/2)-1:int(size_img/2)+int(self.inp_size/2),int(size_img/2)-int(self.inp_size/2)-1:int(size_img/2)+int(self.inp_size/2)])
                   for sn in range(len(random_slices_indx))]


            #==============augment
            #rotation
            if_rotate = np.random.randint(0, 10, len(random_slices_indx)) > 6
            rotate_angle = random.sample(range(-15,15), len(random_slices_indx))
            ASL2= [ndimage.rotate(ASL1[sn],rotate_angle[sn]) if if_rotate[sn] else ASL1[sn] for sn in range(len(random_slices_indx))]
            ASL2= [ASL2[sn][int(ASL2[sn].shape[0]/2)-int(self.inp_size/2)-1:int(ASL2[sn].shape[0]/2)+int(self.inp_size/2),
                   int(ASL2[sn].shape[1]/2)-int(self.inp_size/2)-1:int(ASL2[sn].shape[1]/2)+int(self.inp_size/2)]
                   if ASL2[sn].shape[0]>self.inp_size or ASL2[sn].shape[1]>self.inp_size else ASL2[sn] for sn in range(len(random_slices_indx))]
            if pet is not None:
                PET2= [ndimage.rotate(PET1[sn],rotate_angle[sn]) if if_rotate[sn] else PET1[sn] for sn in range(len(random_slices_indx))]
                PET2 = [PET2[sn][ int(PET2[sn].shape[0] / 2) - int(self.out_size / 2) - 1:int(PET2[sn].shape[0] / 2) + int(
                    self.out_size / 2),
                    int(PET2[sn].shape[1] / 2) - int(self.out_size / 2) - 1:int(PET2[sn].shape[1] / 2) + int(
                        self.out_size / 2)]
                    if PET2[sn].shape[0] > self.out_size or PET2[sn].shape[1] > self.out_size else PET2[sn] for sn in
                    range(len(random_slices_indx))]

            T12= [ndimage.rotate(T11[sn],rotate_angle[sn]) if if_rotate[sn] else T11[sn] for sn in range(len(random_slices_indx))]
            T12 = [T12[sn][ int(T12[sn].shape[0] / 2) - int(self.inp_size / 2) - 1:int(T12[sn].shape[0] / 2) + int(
                self.inp_size / 2),
                    int(T12[sn].shape[1] / 2) - int(self.inp_size / 2) - 1:int(T12[sn].shape[1] / 2) + int(
                        self.inp_size / 2)]
                    if T12[sn].shape[0] > self.inp_size or T12[sn].shape[1] > self.inp_size else T12[sn] for sn in
                    range(len(random_slices_indx))]

            ASL1=ASL2
            if pet is not None:
                PET1=PET2
            T11=T12

            #flip
            if_flip = np.random.randint(0, 10, len(random_slices_indx)) > 5
            ASL2 = [np.fliplr(ASL1[sn]) if if_flip[sn] else ASL1[sn] for sn in
                    range(len(random_slices_indx))]
            if pet is not None:
                PET2 = [np.fliplr(PET1[sn]) if if_flip[sn] else PET1[sn] for sn in
                        range(len(random_slices_indx))]
            T12 = [np.fliplr(T11[sn]) if if_flip[sn] else T11[sn] for sn in
                   range(len(random_slices_indx))]
            ASL1 = ASL2
            if pet is not None:
                PET1 = PET2
            T11 = T12

            #gaussian noise
            mean = 0
            sigma = 0.1 ** 0.5


            if_noise = np.random.randint(0, 10, len(random_slices_indx)) > 5
            ASL2 = [ASL1[sn]+np.random.normal(mean, sigma, (self.inp_size, self.inp_size)).reshape(self.inp_size, self.inp_size)
                    if if_noise[sn] else ASL1[sn] for sn in
                    range(len(random_slices_indx))]
            T12 = [ASL1[sn]+np.random.normal(mean, sigma, (self.inp_size, self.inp_size)).reshape(self.inp_size, self.inp_size)
                   if if_noise[sn] else T11[sn] for sn in
                   range(len(random_slices_indx))]
            ASL1 = ASL2
            T11 = T12

            # if_noise = np.random.randint(0, 10, len(random_slices_indx)) > 6
            #======================


            if len(ASL)==0:
                ASL = ASL1
                PET = PET1
                T1=T11
            else:
                ASL = np.vstack((ASL,ASL1))
                PET = np.vstack((PET,PET1))
                T1 = np.vstack((T1,T11))

        ASL1, PET1, T11=self.shuffle_lists( ASL, PET,T1)

        if self.is_training==1:

            settings.bunch_asl_slices2=ASL1
            settings.bunch_pet_slices2=PET1
            settings.bunch_t1_slices2=T11

        else:

            if len(settings.bunch_pet_slices_vl2)==0:
                settings.bunch_asl_slices2=ASL1
                settings.bunch_pet_slices2=PET1
                settings.bunch_t1_slices2=T11
            else:
                settings.bunch_asl_slices_vl2 = np.vstack((settings.bunch_asl_slices_vl2, ASL1))
                settings.bunch_pet_slices_vl2 = np.vstack((settings.bunch_pet_slices_vl2, PET1))
                settings.bunch_t1_slices_vl2 = np.vstack((settings.bunch_t1_slices_vl2, T11))
        aa = 0
        bb = 0
        if len(settings.bunch_t1_slices2):
            for k in range(len(settings.bunch_pet_slices2)):  # read batches with or without pet patch
                if settings.bunch_pet_slices2[k][0, 0] is not None:
                    aa = aa + 1
                else:
                    bb = bb + 1

        settings.tr_isread=True
        settings.read_patche_mutex_tr.release()
        if len(settings.bunch_t1_slices)!=len(settings.bunch_asl_slices) or len(settings.bunch_asl_slices)!=len(
                settings.bunch_pet_slices)  :
            print('smth wrong')

    #--------------------------------------------------------------------------------------------------------
    def return_patches(self,batch_no,hybrid=0):
        settings.train_queue.acquire()
        asl_slices=[]
        pet_slices=[]
        t1_slices=[]
        list_indx=[]
        if len(settings.bunch_asl_slices)>=batch_no and\
            len(settings.bunch_pet_slices) >= batch_no  :
            # \                        len(settings.bunch_t1_slices) >= batch_no:



            for k in range(len(settings.bunch_asl_slices)): #read batches with or without pet patch
                if hybrid==0: #pet
                    # for k in range(len(settings.bunch_asl_slices)):
                    if settings.bunch_pet_slices[k][0, 0] is not None:
                        list_indx.append(k)

                else: #no pet
                    # for k in range(len(settings.bunch_asl_slices)):
                    if settings.bunch_pet_slices[k][0, 0] is None:
                        list_indx.append(k)
                if len(list_indx) == batch_no:
                    break



            asl_slices= settings.bunch_asl_slices[list_indx]#[0:batch_no]
            pet_slices= settings.bunch_pet_slices[list_indx]#[0:batch_no]
            t1_slices= settings.bunch_t1_slices[list_indx]#[0:batch_no]

            # settings.bunch_asl_slices=np.delete(settings.bunch_asl_slices, range(batch_no), axis=0)
            # settings.bunch_pet_slices=np.delete(settings.bunch_pet_slices, range(batch_no), axis=0)
            # settings.bunch_t1_slices=np.delete(settings.bunch_t1_slices, range(batch_no), axis=0)
            settings.bunch_asl_slices = np.delete(settings.bunch_asl_slices, list_indx, axis=0)
            settings.bunch_pet_slices = np.delete(settings.bunch_pet_slices, list_indx, axis=0)
            settings.bunch_t1_slices = np.delete(settings.bunch_t1_slices, list_indx, axis=0)
            pet_slices = pet_slices[..., np.newaxis]
            asl_slices = asl_slices[..., np.newaxis]
            t1_slices = t1_slices[..., np.newaxis]

        else:
            settings.bunch_asl_slices = np.delete(settings.bunch_asl_slices, range(len(settings.bunch_asl_slices)), axis=0)
            settings.bunch_pet_slices = np.delete(settings.bunch_pet_slices, range(len(settings.bunch_pet_slices)), axis=0)
            settings.bunch_t1_slices = np.delete(settings.bunch_t1_slices, range(len(settings.bunch_t1_slices)), axis=0)
        settings.train_queue.release()
        if len(asl_slices)!=len(pet_slices) :
            print('smth wrong')


        return asl_slices,pet_slices,t1_slices



    #--------------------------------------------------------------------------------------------------------
    def return_patches_validation(self, start,end):
            asl_slices = []
            pet_slices = []
            t1_slices = []

            if (len(settings.bunch_asl_slices_vl) - (end)) >= 0\
                    and (len(settings.bunch_pet_slices_vl) - (end)) >= 0 \
                    and (len(settings.bunch_t1_slices_vl) - (end)) >= 0:



                asl_slices = settings.bunch_asl_slices_vl[start:end]
                pet_slices = settings.bunch_pet_slices_vl[start:end]
                t1_slices = settings.bunch_t1_slices_vl[start:end]

                if len(asl_slices) != len(pet_slices) or len(t1_slices) != len(asl_slices):
                    print('smth wrong')

                pet_slices = pet_slices[..., np.newaxis]
                asl_slices = asl_slices[..., np.newaxis]
                t1_slices = t1_slices[..., np.newaxis]

            return asl_slices, pet_slices,t1_slices
    # -------------------------------------------------------------------------------------------------------

    def return_patches_test(self,start,end):
        asl_slices = []
        pet_slices = []
        t1_slices = []

        if (len(settings.bunch_asl_slices_vl) - (end)) >= 0 \
                and (len(settings.bunch_pet_slices_vl) - (end)) >= 0 \
                and (len(settings.bunch_t1_slices_vl) - (end)) >= 0:

            asl_slices = settings.bunch_asl_slices_vl[start:end]
            pet_slices = settings.bunch_pet_slices_vl[start:end]
            t1_slices = settings.bunch_t1_slices_vl[start:end]

            if len(asl_slices) != len(pet_slices) or len(t1_slices) != len(asl_slices):
                print('smth wrong')

            pet_slices = pet_slices[..., np.newaxis]
            asl_slices = asl_slices[..., np.newaxis]
            t1_slices = t1_slices[..., np.newaxis]

        return asl_slices, pet_slices, t1_slices