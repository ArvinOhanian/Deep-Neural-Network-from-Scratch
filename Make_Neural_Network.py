class Neural_Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_deriv = None
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def call_loss(self, loss, loss_deriv):
        self.loss = loss
        self.loss_deriv = loss_deriv
    
    def predict(self, in_data):
        num_data = len(in_data)
        final_vals = []
        for i in range(num_data):#for each data entry
            out = in_data[i]    #set output value to the data
            for layer in self.layers:#for each layer
                out = layer.forward_propogation(out)#make forward pass based on layer type and set to output
            final_vals.append(out)#save all outputs of forward passes of each data from the set to the final values
            
        return final_vals
    
    def fit(self, input_data, input_labels, num_epochs, learning_rate):
        num_data = len(input_data)
        
        for epoch in range(num_epochs):#for number of predetermined epochs
            error = 0               #each epoch has error reset to 0
            for data in range(num_data):#for each input data
                out = input_data[data]  #store data
                for layer in self.layers:#for every layer in the network
                    out =layer.forward_propogation(out)#do a forward pass with the data
                    
                error +=self.loss(input_labels[data], out)#calculate the loss after going through entire network with each datum and add to total error
                
                back_error = self.loss_deriv(input_labels[data], out)#take derivative of the loss function with inputs of forward prop and label data
                for layer in reversed(self.layers):#propagate through all layers backwards
                    back_error = layer.backpropagation(back_error, learning_rate)#error after backprop 
            
            error /= num_data #avg error
            print('epoch %d/%d   error=%f' % (epoch+1, num_epochs, error))