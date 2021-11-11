import numpy as np

def tangent(x):
    return np.tanh(x)
    
def tangent_deriv(x):
    return 1-np.tanh(x)**2  

def Mean_Squared_Error(actual, predicted):
    out=np.mean(np.power(actual - predicted, 2))
    return out

def Mean_Squared_Error_Deriv(actual, predicted):
    out = 2 * (actual-predicted) / actual.size