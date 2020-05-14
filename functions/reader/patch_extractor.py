import  threading, time
from functions.settings import settings as settings


class _patch_extractor_thread(threading.Thread):
    def __init__ (self, _image_class,
                  img_no,
                  mutex,
                  is_training):
        threading.Thread.__init__(self)

        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())
        self.mutex = mutex
        # number of samples which we read from each bunch of images
        self.sample_no = 50

        if is_training:
            self._image_class=_image_class
        else:
            self._image_class_vl=_image_class
        self.img_no=img_no
        self.is_training=is_training


    def run(self):
        while True:
            with self.pause_cond:
                while self.paused:
                    self.pause_cond.wait()
                try:
                    if self.is_training:

                        self._image_class.read_patche_online_from_image_bunch(self.sample_no, self.img_no)
                    else:

                        if len(settings.bunch_pet_slices_vl) < settings.validation_totalimg_patch:
                            self._image_class_vl.read_patche_online_from_image_bunch_vl(self.sample_no, 5)

                finally:
                    time.sleep(1)






    def pop_from_queue(self):
        return self.queue.get()

    def pause(self):
        # print('pause fill ')

        # If in sleep, we acquire immediately, otherwise we wait for thread
        # to release condition. In race, worker will still see self.paused
        # and begin waiting until it's set back to False
        self.pause_cond.acquire()
        # should just resume the thread
        self.paused = True

    def resume(self):
        # print('resume patch extr ')

        # Notify so thread will wake after lock released
        if self.paused :
            self.pause_cond.notify()
            # Now release the lock
            self.pause_cond.release()
            self.paused = False

    def finish_thread(self):
        self.pause()