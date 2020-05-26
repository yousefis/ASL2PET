import tensorflow as tf
import SimpleITK as sitk
#import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import datetime

class _measure:
    def __init__(self):
        self.eps=0.00001
        print("measurement create object")

    def dice(self, im1, im2):

        if im1.shape != im2.shape:
            raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

        # Compute Dice coefficient
        intersection = np.logical_and(im1, im2)

        return 2. * intersection.sum() / (im1.sum() + im2.sum()+self.eps)

