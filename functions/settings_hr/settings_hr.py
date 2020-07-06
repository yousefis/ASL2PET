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
from threading import Lock

def init():
    global bunch_t1_slices2, bunch_pet_slices,bunch_asl_slices2, bunch_pet_slices2,mutex,mutex2,bunch_t1_slices_vl2,bunch_t1_slices
    global bunch_asl_slices_vl, bunch_pet_slices_vl, bunch_pet_slices_vl2, patch_count,bunch_t1_slices_vl,bunch_asl_slices_vl2
    global bunch_t1_slices,bunch_t1_slices2,bunch_t1_slices_vl,bunch_t1_slices_vl2,bunch_asl_slices
    global train_queue, read_patche_mutex_tr,read_patche_mutex_vl,tr_isread,vl_isread
    global queue_isready_vl, validation_totalimg_patch, validation_patch_reuse, read_vl_offline, read_off_finished, epochs_no
    queue_isready_vl = False



    validation_totalimg_patch = 539#5*17 #number of images should be read for validation: here we have 1subject for validation containing pets and asl scans, every 17 slices
    read_vl_offline = False
    read_off_finished = False
    epochs_no=0
    patch_count=0
    mutex= Lock()
    mutex2= Lock()
    read_patche_mutex_tr= Lock()
    read_patche_mutex_vl= Lock()
    train_queue=Lock()
    bunch_t1_slices=[]

    tr_isread=True
    vl_isread = True
    bunch_asl_slices2=[]
    bunch_pet_slices2=[]
    bunch_t1_slices2=[]

    bunch_asl_slices = []
    bunch_pet_slices = []
    bunch_t1_slices = []

    bunch_asl_slices_vl=[]
    bunch_pet_slices_vl=[]
    bunch_t1_slices_vl=[]

    bunch_asl_slices_vl2=[]
    bunch_pet_slices_vl2=[]

    bunch_t1_slices_vl2 = []
    validation_patch_reuse = []