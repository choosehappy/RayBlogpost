import glob

import numpy as np
import openslide
import ray
from skimage import color, img_as_ubyte, io
from skimage.filters import rank
from skimage.morphology import disk

from openslide import OpenSlideError


@ray.remote
def load_and_run_minimumPixelIntensityNeighborhoodFiltering(fname):
    try:
        osh = openslide.OpenSlide(fname)
        dim = 2
        img = osh.read_region((0, 0), dim, osh.level_dimensions[dim])
        img = np.asarray(img)[:, :, 0:3]

        disk_size = 5
        threshold = 210

        img = color.rgb2gray(img)
        img = (img * 255).astype(np.uint8)
        selem = disk(disk_size)

        imgfilt = rank.minimum(img, selem)
        imgout = imgfilt > threshold
        io.imsave(
            f"./opt/{fname.replace('/data/','').replace('.svs','_mask.png')}",
            img_as_ubyte(imgout),
        )
        return imgout.mean()

    except OpenSlideError as err:
        print(err)
    except Exception as err:
        return -1


files = glob.glob("./data/*.svs")

futures = [
    load_and_run_minimumPixelIntensityNeighborhoodFiltering.remote(f) for f in files
]

results = [ray.get(future) for future in futures]
print(results)
