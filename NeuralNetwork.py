from Layer import Layer
class NeuralNetwork:
    __slots__ = ['inputs','outputs','layers','first_layer']
    def __init__(self,inputs:int,outputs:int,hidden_layers:[int]):# every hidden_layers's  int is a number of neurons in layer
        self.layers:[int]=hidden_layers+[outputs]
        self.first_layer=Layer(self.layers,inputs)





    def train(self,x:[[float]],y:[[float]],epochs:int):

        for _ in range(epochs):
            for data_set,answers in zip(x,y):

                self.first_layer.train(data_set,answers)


    def calculate(self,x) -> [float]:
        return self.first_layer.forward(x)



    def set_weights(self,weights:[[[float]]]):
        self.first_layer.set_weights(weights)

    def get_weights(self):
        return self.first_layer.get_weights()