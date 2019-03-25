# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 23:15:02 2019

@author: ljp
"""

import numpy as np
from scipy.stats import multivariate_normal
import random
from scipy.ndimage import convolve





##### Function that creates plans from parameters #####

def unet_create_plans(number_of_plans, plan_size, num_mean, num_std, info_neuron, coord_threshold):
    
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
    - coord_temp: coordinates of all neurons, with associated plan ((number_of_plans, plan_size[0], plan_size[1]) matrix)
    """
    
    # Retriving information on neurons
    neu_mean, neu_std, cor_mean, cor_std, xvar_mean, xvar_std, yvar_mean, yvar_std = info_neuron
    
    # Creating initial set of plans
    plans_temp = np.zeros((number_of_plans, plan_size[0], plan_size[1]))
    # Defining coordinates vector
    coord_temp = np.zeros((number_of_plans, plan_size[0], plan_size[1]))
    
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
            # Replacing in final plan
            plan_temp = valexp.reshape(13, 13) 
            plans_temp[i, (randx-7):(randx+6), (randy-7):(randy+6)] = plan_temp
            # Adding neuron detection in coord_temp
            coord_temp[i, (randx-7):(randx+6), (randy-7):(randy+6)] = 1 * (plan_temp > neu_mean/coord_threshold)
            
    return plans_temp, coord_temp





##### Function that stacks plans from create_plans function #####

def unet_stack_plans(nstacks, plans_dataset, coord_dataset, cropped_size, stack_mean, stack_std, noise_mean, noise_std):
    
    """Function that randomly stack plans given as inputs.
    
    %%%%% INPUTS %%%%%
    - nstacks: final number of stacked plans (integer)
    - plans_dataset: plans dataset obtained from unet_create_plans (3d matrix)
    - coord_dataset: coordinates dataset obtained from unet_create_plans (3d matrix)
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
    coord_stack = np.zeros((nstacks, cropped_size, cropped_size))
    
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
        coords_temp = np.zeros((stack_f, cropped_size, cropped_size))
        # Chosing plans to use
        plan_use = np.floor(nstacks_old*np.random.rand(stack_f)).astype(int)
        # Second for loop for each plan
        for j in range(stack_f):
            # Loading plan and coordinates
            plan_temp = plans_dataset[plan_use[j], :, :]
            coord_temp = coord_dataset[plan_use[j], :, :]
            # Cropping
            xcrop_init = random.randint(0, xcrop) #np.random.random_integers(0, xcrop)
            ycrop_init = random.randint(0, ycrop) #np.random.random_integers(0, ycrop)
            plan_temp = plan_temp[xcrop_init:(xcrop_init+cropped_size), ycrop_init:(ycrop_init+cropped_size)]  
            coord_temp = coord_temp[xcrop_init:(xcrop_init+cropped_size), ycrop_init:(ycrop_init+cropped_size)]         
            # Rotating
            rotnum = random.randint(0, 3) #np.random.random_integers(0, 3)
            for k in range(rotnum):
                plan_temp = plan_temp.T[::-1, :]
                coord_temp = coord_temp.T[::-1, :]
            # Adding to temporary stack
            plans_temp[j, :, :] = plan_temp
            coords_temp[j, :, :] = coord_temp
        # Adding noise plan
        plan_noise = noise_mean + np.abs(noise_std*np.random.randn(cropped_size, cropped_size))
        plans_temp[-1, :, :] = convolve(plan_noise, filter_noise, mode="constant") # convolution to filter
        # Merging stacks into stack_plan
        stack_plan_temp = np.max(plans_temp, axis=0)
        stack_plan_temp[stack_plan_temp < 0] = 0
        plans_stack[i, :, :] = stack_plan_temp
        # Deleting doublons in stack_coord
        argmax_plan_temp = np.argmax(plans_temp, axis=0) + 1
        coord_stack[i, :, :] = argmax_plan_temp * np.max(coords_temp, axis=0)
        
    # Returning values
    return plans_stack, coord_stack