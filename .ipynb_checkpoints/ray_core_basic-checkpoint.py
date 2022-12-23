#!/usr/bin/env python
# coding: utf-8
# %%
import glob, cv2


# %%
fnames = glob.glob("/opt/data/*.bmp")


# %%
#Single threaded approach for reading, rotating an image, saving it, and returning the mean
def rotateImage90(fname):
    img = cv2.imread(fname)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(fname.replace('.bmp','_rot90.bmp'),img)
    return img.mean()


# %%
# %%time
mean_vals = [rotateImage90(f) for f in fnames]
print(mean_vals)

#On my machine this takes ~400ms, not bad!


# %%
#---- Multiprocessing with Ray

import ray

@ray.remote  #convert function to ray multi processing function by simply adding this single annotation
def ray_rotateImage90(fname):
    img = cv2.imread(fname)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(fname.replace('.bmp','_rot90.bmp'),img)
    return img.mean()


# %%
# %%time
ray.init() #or ray.init(address='ray://[IP_OF_HEAD_NODE]:10001') if going distributed
ray.available_resources()


# %%
# %%time
#now look at how long it takes using ray
futures = [ray_rotateImage90.remote(f) for f in fnames]
mean_vals=ray.get(futures)
print(mean_vals)


# %%
#we can see the runtime is now significantly faster, on my laptop 300ms 

#so the two main takeaways:
#1. ray functions can be created by simply decorating them as ray functions
#2. it takes a long time to initialize a ray cluster, but after initilaized, it can be used easily

