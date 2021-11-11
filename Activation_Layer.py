from Layer import Layer


class ActivationLayer(Layer):
    def __init__(self, act, act_deriv):
        self.active = act
        self.active_deriv = act_deriv
        
    def forward_propagation(self, data_in):
        self.input = data_in
        self.output = self.active(self.input)#output is the activation function with input data
        return self.output
    
    def backpropagation(self, out_err, learning_rate):
        in_err = self.active_deriv(self.input) * out_err
        return in_err
    
    