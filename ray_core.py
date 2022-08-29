import glob
import numpy as np
import openslide
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte
import logging
import os
import numpy as np
from skimage import io, color, img_as_ubyte
from distutils.util import strtobool
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
from sklearn.cluster import KMeans
from skimage import exposure


@ray.remote
def load_and_run_minimumPixelIntensityNeighborhoodFiltering(fname):
    try:
        osh=openslide.OpenSlide(fname)
        dim=2
        img=osh.read_region((0,0),dim,osh.level_dimensions[dim])
        img= np.asarray(img)[:, :, 0:3]

        disk_size =  5
        threshold = 210

        img = color.rgb2gray(img)
        img = (img * 255).astype(np.uint8)
        selem = disk(disk_size)

        imgfilt = rank.minimum(img, selem)
        imgout =  imgfilt > threshold
        io.imsave(f"/opt/{fname.replace('/data/','').replace('.svs','_mask.png')}",img_as_ubyte(imgout))
        return imgout.mean()
    except:
        return -1
