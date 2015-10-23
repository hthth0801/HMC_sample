# -*- coding: utf-8 -*-
"""
Created on Sat Oct 03 11:44:30 2015
load various datasets
@author: tian
"""

import numpy as np
import array
#import pylab
from os.path import join, isfile
from os import listdir

rng = np.random.RandomState(123)
def read_vanhateren_images (n_imgs=5):
    """
    read the specified number images from van-hateren imc dataset
    returns the list of readed images
    """
    folder_name = r'D:\VanHateren\vanhateren_imc' # change this to point to the directory which holds the van hateren data
   # files = listdir(folder_name)
    onlyfiles = [ f for f in listdir(folder_name) if isfile(join(folder_name,f)) ]
    imgs = []
    for i in range(n_imgs):
        filename = join(folder_name, onlyfiles[i])
        with open(filename, 'rb') as handle:
            s = handle.read()
        arr = array.array('H', s)
        arr.byteswap()
        img_i = np.array(arr, dtype='uint16').reshape(1024, 1536)
        imgs.append(img_i)  
    return imgs
        #pylab.imshow(img)
    #pylab.show()
        
def get_random_patches(images, n_patches, patch_x, patch_y):
    """
    we randomly extract patches from images. Similar to the skdata
    input images are np array, [#images]*[ncols]*[nrows]
    """
    n_images = images.shape[0]
    img_c = images.shape[1]
    img_r = images.shape[2]
    r_images = rng.randint(n_images, size = n_patches)
    r_x = rng.randint(img_c-patch_x+1, size = n_patches)
    r_y = rng.randint(img_r-patch_y+1, size = n_patches)
    patches_list = []
    for image_i, x_i, y_i in zip(r_images, r_x, r_y):
        patch_i = images[image_i, x_i:(x_i + patch_x), y_i:(y_i + patch_y)]
        patches_list.append(patch_i)
    
    patches_np = np.asarray(patches_list, dtype=images.dtype)   
    return patches_np  
    
def log_center_transform(X):
    """
    input: X [n_sample]*[n_dim], array-like
    do the log-transformation and centering
    """
    X = np.asarray(X, dtype='float32')
    X[X<1.] = 1.
    X = np.log(X)
    # center
    X = X - np.mean(X, axis=1).reshape((-1,1))
    return X

def PCA_ZCA_whiten_transform(X, symmetric = True):
    """
    perform PCA whitening or ZCA whitening
    """
    X = np.asarray(X, dtype='float32')
    #center
    X = X - np.mean(X, axis=1).reshape((-1,1))
    C = np.dot(X.T, X) / X.shape[0]
    EigVal, EigVec = np.linalg.eigh(C)
    mval = np.max(np.real(EigVal))
    max_ratio = 1e4
    tol = mval / max_ratio
    #ngd = np.nonzero(np.real(EigVal) > mval/max_ratio)
    if symmetric:
        ngd = np.nonzero(np.real(EigVal) > tol)[0]
        P = (np.real(EigVal[ngd])**(-0.5)).reshape((-1,1)) * EigVec[:, ngd].T # [reduced dim]*[dim]
        P = np.dot(EigVec[:, ngd], P)
    else:
        EigVal[EigVal <= tol] = 1.
        P = EigVal.reshape((-1,1)) * (EigVec.T)
    X = np.dot(X, P.T)
    return X, P.T      
          
    
def load_van_hateren(n_imgs=5, n_patches=100000, patch_x=10, patch_y=10):
    """
    first get raw patches, then do the necessary preprocessing
    """
    images = np.asarray(read_vanhateren_images(n_imgs))
    patches_raw = get_random_patches(images, n_patches, patch_x, patch_y)
    patches_float = patches_raw.astype('float32')
    patches = patches_float.reshape(len(patches_float), -1)
    # do preprocessing (log-tranformation and centering)
    patches = log_center_transform(patches)
    # do PCA elimination to get rid of the degenerated dimensions
    patches, W_X = PCA_ZCA_whiten_transform(patches, False)
    return patches, W_X

def load_pixel_sparse(n_imgs=5, n_patches=100000, patch_x=4, patch_y=4):
    """
    Generate pixel-sparse training data.
    """
    n = np.random.randn(n_patches, patch_x*patch_y)
    patches_unnorm = n**3
    patches = patches_unnorm / np.std(patches_unnorm)
    W_X = np.eye(patch_x*patch_y)
    # DEBUG why is this different from what's expected of load_van_hateren
    #return patches, W_X
    return patches
        
"""      
patches = load_van_hateren()
images = np.asarray(read_vanhateren_images(5))
images_extend = images[:, :, :, None]
filename = join(folder_name, onlyfiles[0])
s = open(filename, 'rb').read()
arr = array.array('H', s)
arr.byteswap()
img = np.array(arr, dtype='uint16').reshape(1024, 1536)
pylab.imshow(img)
image = np.asarray(img)
"""
