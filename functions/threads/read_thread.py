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
import threading, time
import numpy as np
from functions.settings import settings as settings


class read_thread(threading.Thread):
    '''
    This class reads the patches from a bunch of images in a parallel way
    '''
    def __init__ (self,_fill_thread,mutex,validation_sample_no=0,is_training=1):
        threading.Thread.__init__(self)
        self._fill_thread=_fill_thread
        self.mutex=mutex

        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.is_training=is_training
        self.validation_sample_no=validation_sample_no
    def run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                try:
                    if self.is_training==1:
                        # if it is training
                        if len(settings.bunch_asl_slices)==0 and len(settings.bunch_pet_slices)==0 and len(
                                settings.bunch_t1_slices)==0:
                            settings.train_queue.acquire()
                            settings.bunch_asl_slices= settings.bunch_asl_slices2.copy()
                            settings.bunch_pet_slices= settings.bunch_pet_slices2.copy()
                            settings.bunch_t1_slices= settings.bunch_t1_slices2.copy()

                            settings.bunch_asl_slices2 = []
                            settings.bunch_pet_slices2 = []
                            settings.bunch_t1_slices2 = []
                            settings.train_queue.release()
                            self._fill_thread.resume()
                        elif not len(settings.bunch_asl_slices) == 0 and not len(
                                settings.bunch_pet_slices) == 0 and not len(settings.bunch_t1_slices) == 0:
                            #if it is validating
                            if len(settings.bunch_asl_slices2) and len(settings.bunch_pet_slices2)and len(
                                    settings.bunch_t1_slices2):
                                settings.train_queue.acquire()
                                settings.bunch_asl_slices = np.vstack((settings.bunch_asl_slices,
                                                                       settings.bunch_asl_slices2))
                                settings.bunch_pet_slices = np.vstack((settings.bunch_pet_slices,
                                                                       settings.bunch_pet_slices2))
                                settings.bunch_t1_slices = np.vstack((settings.bunch_t1_slices,
                                                                      settings.bunch_t1_slices2))
                                settings.bunch_asl_slices2=[]
                                settings.bunch_pet_slices2=[]
                                settings.bunch_t1_slices2 = []
                                settings.train_queue.release()
                                self._fill_thread.resume()

                    else:
                        if len(settings.bunch_asl_slices_vl) > settings.validation_totalimg_patch:
                            del settings.bunch_asl_slices_vl2
                            del settings.bunch_pet_slices_vl2
                            del settings.bunch_t1_slices_vl2
                            break
                        if ((len(settings.bunch_pet_slices_vl) == 0) \
                                &(len(settings.bunch_pet_slices_vl) == 0)\
                                &(len(settings.bunch_t1_slices_vl) == 0)\
                                &(len(settings.bunch_asl_slices_vl2) > 0)\
                                &(len(settings.bunch_t1_slices_vl2) > 0)\
                                &(len(settings.bunch_pet_slices_vl2) > 0)):
                            settings.bunch_asl_slices_vl = settings.bunch_asl_slices_vl2
                            settings.bunch_asl_slices_vl2 = []

                            settings.bunch_pet_slices_vl = settings.bunch_pet_slices_vl2
                            settings.bunch_pet_slices_vl2 = []

                            settings.bunch_t1_slices_vl = settings.bunch_t1_slices_vl2
                            settings.bunch_t1_slices_vl2 = []
                            print('settings.bunch_asl_slices_vl lEN: %d' % (len(settings.bunch_asl_slices_vl)))
                        elif ((len(settings.bunch_pet_slices_vl) > 0) \
                                &(len(settings.bunch_pet_slices_vl) > 0)\
                                &(len(settings.bunch_t1_slices_vl) > 0)\
                                &(len(settings.bunch_pet_slices_vl2) > 0)\
                                &(len(settings.bunch_pet_slices_vl2) > 0)\
                                &(len(settings.bunch_t1_slices_vl2) > 0)):
                            settings.bunch_asl_slices_vl = np.vstack((settings.bunch_asl_slices_vl,
                                                                      settings.bunch_asl_slices_vl2))
                            settings.bunch_asl_slices_vl2 = []

                            settings.bunch_pet_slices_vl = np.vstack((settings.bunch_pet_slices_vl,
                                                                      settings.bunch_pet_slices_vl2))
                            settings.bunch_pet_slices_vl2 = []

                            settings.bunch_t1_slices_vl = np.vstack((settings.bunch_t1_slices_vl, settings.bunch_t1_slices_vl2))
                            settings.bunch_t1_slices_vl2 = []
                            print('settings.bunch_asl_slices_vl lEN2: %d' % (len(settings.bunch_asl_slices_vl)))

                        if len(settings.bunch_pet_slices_vl)<self.validation_sample_no:
                            if self._fill_thread.paused==True:
                                self._fill_thread.resume()
                        else:
                            self.finish_thread()
                finally:
                    time.sleep(1)




    def pause(self):
        # print('pause read ')

        # If in sleep, we acquire immediately, otherwise we wait for thread
        # to release condition. In race, worker will still see self.paused
        # and begin waiting until it's set back to False

        self.pause_cond.acquire()
        # should just resume the thread
        self.paused = True

    def resume(self):
        # print('resume read ')
        if self.paused:
            # Notify so thread will wake after lock released
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()
            self.paused = False

    def finish_thread(self):
        self.pause()

