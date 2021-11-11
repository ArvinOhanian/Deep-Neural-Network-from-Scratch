from Layer import Layer
import numpy as np

class Fully_Connected_Layer(Layer):
    
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5 #make an random array(nums between -.5 and .5) of size input X output since each input will have output
                                                                     #number of weights thateach connect to the next layer
        self.bias = np.random.rand(1, output_size) - 0.5#input*weights = [1, output_size] then add to bias of same dimensions
    
    def forward_propagation(self, input_vals):
        self.input = input_vals#[1, in_size]
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
                                                   
    def backpropagation(self, out_err, learning_rate):
        in_err = np.dot(out_err, self.weights.T)# out_err[1, out_size] * weights.T[out_size, in_size] = in_err[1, in_size]
        weights_err = np.dot(self.input.T, out_err)#input.T[in_size, 1] * out_err[1, out_size] = weights_err[in_size, out_size]
        self.weights -= weights_err * learning_rate#update weights
        self.bias -= out_err * learning_rate#update biases
        return in_err#return input error
        