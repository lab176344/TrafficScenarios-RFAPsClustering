# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 11:03:54 2020

@author: balasubramanian
"""

import scipy.io as io
import numpy as np
from einops import rearrange
xTest   = io.loadmat('Simulated_Test.mat')['m_xTest']
yTest   = io.loadmat('Simulated_Test.mat')['m_yTest']
xTrain1 = io.loadmat('Simulated_Train1.mat')['m_xTrain1']
xTrain2 = io.loadmat('Simulated_Train2.mat')['m_xTrain2']
xTrain3 = io.loadmat('Simulated_Train3.mat')['m_xTrain3']
xTrain4 = io.loadmat('Simulated_Train4.mat')['m_xTrain4']
yTrain = io.loadmat('Simulated_Train4.mat')['m_yTrain']


xTrain = np.concatenate((xTrain1, xTrain2, xTrain3, xTrain4),axis=3)

xTrain = rearrange(xTrain, 'h w d b -> b h w d')
yTrain = rearrange(yTrain, '1 b -> b')

xTest = rearrange(xTest, 'h w d b -> b h w d')
yTest = rearrange(yTest, '1 b -> b')




