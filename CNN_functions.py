# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:50:14 2019

@author: Hippolyte Moulle
"""

import numpy as np
from scipy.stats import multivariate_normal
import random
from scipy.ndimage import convolve





##### Function that creates plans from parameters #####

def create_plans(number_of_plans, plan_size, num_mean, num_std, info_neuron):
    
    """Function that creates stack of plans from parameters.
    
    %%%%% INPUTS %%%%%
    - number_of_plans: final number of stacked plans in output (integer)
    - plan_size: size of each plan in pixels (vector of 2 integers)
    - num_mean: mean number of neurons per plan (integer)
    - num_std: standard deviation of number of neurons per plan (integer)
    - info_neuron: series of 8 floats describing each neuron, comprising:
            - neu_mean: mean intensity
            - neu_std: standard deviation of intensity
            - cor_mean: mean covariance (inclinaison)
            - cor_std: standard deviation of covariance
            - xvar_mean: mean variance in the x-axis direction
            - xvar_std: standard deviation of the variance in the x-axis direction
            - yvar_mean: mean variance in the y-axis direction
            - yvar_std: standard deviation of the variance in the y-axis direction
            
    %%%%% OUTPUTS %%%%%
    - plans_temp: plans with neurons stacked along first dimension ((number_of_plans, plan_size[0], plan_size[1]) matrix)
    - coord_temp: coordinates of all neurons, with associated plan ((number of neurons, 3) matrix)
    """
    
    # Retriving information on neurons
    neu_mean, neu_std, cor_mean, cor_std, xvar_mean, xvar_std, yvar_mean, yvar_std = info_neuron
    
    # Creating initial set of plans
    plans_temp = np.zeros((number_of_plans, plan_size[0], plan_size[1]))
    # Defining coordinates vector
    coord_temp = np.zeros((0, 3))
    
    # Creating Xfield and Yfield vectors
    Xfield, Yfield = np.mgrid[-6:7, -6:7]
    Xfield = Xfield.flatten()[:, np.newaxis]
    Yfield = Yfield.flatten()[:, np.newaxis]
    
    # Defining number of neurons per plan
    numfin = np.round_(num_mean + num_std*np.random.randn(number_of_plans))
    # Loop on each plan
    for i in range(number_of_plans):
        # Loop on each potential neuron in plan i
        for j in range(int(numfin[i])):
            # Finding available space for neuron
            finding = -1
            numround = 0
            numlim = 50
            while finding == -1 and numround < numlim:
                randx = int(np.round_(np.random.rand() * (plan_size[0] - 15)) + 7)
                randy = int(np.round_(np.random.rand() * (plan_size[1] - 15)) + 7)
                anafield = plans_temp[i, (randx-7):(randx+6), (randy-7):(randy+6)]
                if np.count_nonzero(anafield) == 0:
                    break
                numround += 1
            if numround == numlim:
                continue
            # Defining neuron's information
            matcorval = cor_mean + cor_std * np.random.randn()
            matrixstd = np.array([[xvar_mean + xvar_std*np.random.randn(), matcorval], 
                                 [matcorval, yvar_mean + yvar_std*np.random.randn()]])
            if np.linalg.det(matrixstd) < 0:
                continue
            # Making matrix with neuron's information
            numval = (neu_mean + neu_std*np.random.randn()) * np.sqrt((2*np.pi)**2 * np.linalg.det(matrixstd))
            mvn = multivariate_normal(mean = np.array([0, 0]), cov = matrixstd)
            valexp = numval * mvn.pdf(np.hstack((Xfield, Yfield)))
            if numval <= 0:
                continue
            # Adding neuron coordinates in coord_temp
            coord_temp = np.vstack((coord_temp, np.array([i, randx-1, randy-1])))
            # Replacing in final plan
            plans_temp[i, (randx-7):(randx+6), (randy-7):(randy+6)] = valexp.reshape(13, 13) 
            
    return plans_temp, coord_temp





##### Function that stacks plans from create_plans function #####

def stack_plans(nstacks, plans_dataset, coord_dataset, cropped_size, stack_mean, stack_std, noise_mean, noise_std):
    
    """Function that randomly stack plans given as inputs.
    
    %%%%% INPUTS %%%%%
    - nstacks: final number of stacked plans (integer)
    - plans_dataset: plans dataset obtained from create_plans (3d matrix)
    - coord_dataset: coordinates dataset obtained from create_plans (2d matrix)
    - cropped_size: final size of plans after cropped (integer)
    - stack_mean: mean number of plans stacked for one final plan (integer)
    - stack_std: standard deviation of number of plans stacked for one final plan (integer)
    - noise_mean: mean of noise plan added to stacked plans (float)
    - noise_std: standard deviation of noise plan added to stacked plans (float)
    
    %%%%% OUTPUTS %%%%%
    - plans_stack: final plans stacked along first dimension ((nstacks, cropped_size, cropped_size) matrix)
    - coord_stack: coordinates of all neurons, with associated final plan ((number of neurons, 3) matrix)
    """
    
    # First defining final matrices
    plans_stack = np.zeros((nstacks, cropped_size, cropped_size))
    coord_stack = np.zeros((0, 3))
    
    # Number of stacks in plans_dataset
    nstacks_old = plans_dataset.shape[0]
    
    # Parameters for cropping
    xcrop = plans_dataset.shape[1] - cropped_size
    ycrop = plans_dataset.shape[2] - cropped_size
    
    # Filter for noise, will be convolved to have continuous noise
    filter_shape = 5
    filter_noise = np.zeros((filter_shape, filter_shape))
    for i in range(filter_shape):
        for j in range(filter_shape):
            radius_temp = np.sqrt((i-(filter_shape-1)/2)**2 + (j-(filter_shape-1)/2)**2)
            if radius_temp <= 1:
                filter_noise[i, j] = 1
            elif radius_temp <= 2:
                filter_noise[i, j] = 0.5  
                
    # Launching loop
    for i in range(nstacks):
        # Selecting number of stacks
        stack_f = int(np.round_(stack_mean + stack_std*np.random.randn()))
        if stack_f <= 0:
            stack_f = 1
        # Creating temporary plan
        plans_temp = np.zeros((stack_f+1, cropped_size, cropped_size))
        # Creating a temporaty coordinates matrix
        coords_temp = np.zeros((0, 2))
        # Chosing plans to use
        plan_use = np.floor(nstacks_old*np.random.rand(stack_f)).astype(int)
        # Second for loop for each plan
        for j in range(stack_f):
            # Loading plan and coordinates
            plan_temp = plans_dataset[plan_use[j], :, :]
            coord_temp = coord_dataset[coord_dataset[:, 0] == plan_use[j], 1:3]
            # Cropping
            xcrop_init = random.randint(0, xcrop) #np.random.random_integers(0, xcrop)
            ycrop_init = random.randint(0, ycrop) #np.random.random_integers(0, ycrop)
            coord_temp = coord_temp - np.array([[xcrop_init, ycrop_init]])
            coord_temp = coord_temp[coord_temp[:, 0] < cropped_size, :]
            coord_temp = coord_temp[coord_temp[:, 0] >= 0, :]
            coord_temp = coord_temp[coord_temp[:, 1] < cropped_size, :]
            coord_temp = coord_temp[coord_temp[:, 1] >= 0, :]
            plan_temp = plan_temp[xcrop_init:(xcrop_init+cropped_size), ycrop_init:(ycrop_init+cropped_size)]          
            # Rotating
            rotnum = random.randint(0, 3) #np.random.random_integers(0, 3)
            for k in range(rotnum):
                plan_temp = plan_temp.T[::-1, :]
                coord_temp = np.vstack((cropped_size-1-coord_temp[:, 1], coord_temp[:, 0])).T 
            # Adding to temporary stack
            plans_temp[j, :, :] = plan_temp
            coords_temp = np.vstack((coords_temp, coord_temp))
        # Adding noise plan
        plan_noise = noise_mean + np.abs(noise_std*np.random.randn(cropped_size, cropped_size))
        plans_temp[-1, :, :] = convolve(plan_noise, filter_noise, mode="constant") # convolution to filter
        # Merging stacks into stack_plan
        stack_plan_temp = np.max(plans_temp, axis=0)
        stack_plan_temp[stack_plan_temp < 0] = 0
        plans_stack[i, :, :] = stack_plan_temp
        # Deleting doublons in stack_coord
        temp_unique = np.unique(coords_temp, axis=0)
        temp_unique = np.hstack((i*np.ones((temp_unique.shape[0], 1)), temp_unique))
        coord_stack = np.vstack((coord_stack, temp_unique))
        
    # Returning values
    return plans_stack, coord_stack





##### Function that builds training and test sets #####
    
def build_dataset(plans_in, coord_in, training_proportion):
    
    """Function that creates dataset based on stacked plans and coordinates.
    
    %%%%% INPUTS %%%%%
    - plans_in: stacked plans, obtained from stack_plans function (3d matrix)
    - coord_in: neurons coordinates, obtained from stack_plans function (2d matrix)
    - training_proportion: training set proportion of whole set (float in ]0, 1[)
    
    %%%%% OUTPUTS %%%%%
    - Xtrain: training set of plans stacked along first dimension (3d matrix)
    - Ytrain: training labels the length equal to number of pixels, stacked along first dimension (2d matrix)
    - Xtest: test set of plans stacked along first dimension (3d matrix)
    - Ytest: test labels the length equal to number of pixels, stacked along first dimension (2d matrix)
    """
    
    # Getting size of images
    crop_size = plans_in.shape[1]
    # Building training and test sets
    Xtrain = np.zeros((0, crop_size, crop_size))
    Ytrain = np.zeros((0, crop_size*crop_size))
    Xtest = np.zeros((0, crop_size, crop_size))
    Ytest = np.zeros((0, crop_size*crop_size))
    
    # For-loop on number of examples
    lensk = plans_in.shape[0]
    test_switch = int(np.round_(training_proportion*lensk))
    for i in range(lensk):
        coord_in_temp = coord_in[coord_in[:, 0] == i, 1:3]
        Ytemp = np.zeros((crop_size, crop_size))
        Ytemp[coord_in_temp[:, 0].astype(int), coord_in_temp[:, 1].astype(int)] = 1
        if i < test_switch:
            Ytrain = np.vstack((Ytrain, Ytemp.reshape(1, crop_size*crop_size)))
            Xtrain = np.concatenate((Xtrain, plans_in[i, :, :][np.newaxis, :, :]), axis=0)
        else:
            Ytest = np.vstack((Ytest, Ytemp.reshape(1, crop_size*crop_size)))
            Xtest = np.concatenate((Xtest, plans_in[i, :, :][np.newaxis, :, :]), axis=0)
            
    # Returning sets
    return Xtrain, Ytrain, Xtest, Ytest





##### Function that 'compresses' plans and coord #####

def compress_plans(plans_in, coord_in, compression):
    
    """Function that creates new dataset, compressing pixels.
    
    %%%%% INPUTS %%%%%
    - plans_in: stacked plans, obtained from stack_plans function (3d matrix)
    - coord_in: neurons coordinates, obtained from stack_plans function (2d matrix)
    - compression: number of pixels averaged in each direction for each new pixel (integer)
    
    %%%%% OUTPUTS %%%%%
    - plans_out: new plan (3d matrix)
    - coord_out: new coordinates. By convention, neurons centers are attributed to lower pixel if not integer (2d matrix)
    """
    
    # Making sure compression is an integer
    compression = int(compression)
    
    # Building new plans and coordinates
    dataset_size = plans_in.shape[0]
    crop_size = plans_in.shape[1]
    blow = int(np.ceil(crop_size/compression))
    plans_out = np.zeros((dataset_size, blow, blow))
    coord_out = np.hstack((coord_in[:, 0][:, np.newaxis], np.floor(coord_in[:, 1:3] / compression)))
    coord_out = np.unique(coord_out, axis=0)
    
    # For-loop on each 
    for k in range(dataset_size):
        # All new pixels except last row and column
        for i in range(blow-1):
            for j in range(blow-1):
                plans_out[k, i, j] = np.mean(plans_in[k, compression*i:compression*(i+1), compression*j:compression*(j+1)])
        # Last row and column
        for i in range(blow-1):
            plans_out[k, i, blow-1] = np.mean(plans_in[k, compression*i:compression*(i+1), compression*(blow-1):-1])
        for j in range(blow-1):
            plans_out[k, i, blow-1] = np.mean(plans_in[k, compression*(blow-1):-1, compression*j:compression*(j+1)])
        plans_out[k, blow-1, blow-1] = np.mean(plans_in[k, compression*(blow-1):-1, compression*(blow-1):-1])
        
    # Returning outputs
    return plans_out, coord_out