import  threading, time

import settings.settings as settings
class fill_thread(threading.Thread):
    def __init__ (self, data,_image_class,
                  mutex,is_training,patch_extractor):
        """
            Thread for moving images to RAM.

            This thread moves the images to RAM for train and validation process simultaneously fot making this process co-occurrence.

            Parameters
            ----------
            arg1 : int
                Description of arg1
            arg2 : str
                Description of arg2

            Returns
            -------
            nothing


        """
        threading.Thread.__init__(self)
        self.data=data
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self._image_class=_image_class
        self.mutex=mutex
        self.is_training=is_training
        self.patch_extractor=patch_extractor

        self.Kill=False


    def run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                try:
                    # for validation
                    if self.is_training==0:
                        if len(settings.bunch_pet_slices_vl) >= settings.validation_totalimg_patch:
                            break
                        if settings.vl_isread == False:
                            continue
                        self._image_class.read_bunch_of_images_vl( settings.validation_totalimg_patch)
                        self.patch_extractor.resume()
                    # for train
                    else:
                        if settings.tr_isread == False:
                            continue
                        self._image_class.read_bunch_of_images()
                        self.patch_extractor.resume()
                finally:
                    #thread sleeps for 1sec
                    time.sleep(1)





    def pop_from_queue(self):
        return self.queue.get()
    def kill_thread(self):
        self.Kill=True

    def pause(self):
        self.pause_cond.acquire()
        # should just resume the thread
        self.paused = True

    def resume(self):
        if self.paused :
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()
            self.paused = False

