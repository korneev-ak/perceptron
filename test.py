from NeuralNetwork import NeuralNetwork
import math


if __name__ == '__main__':
    network = NeuralNetwork(1,1,[15],"sigmoid")
    #arr=[[[0.5,-0.1,0.2],[0.3,-0.23,0.43]],[[0.33,-0.4,],[-0.5,0.2]],[[-0.49,0.39]]]
    #network.set_weights(arr)
    #network.train([[1,0,1],[0,1,1],[1,0,0],[1,1,1],[0,0,1],[0,1,0],[0,0,0]],[[1],[0],[1],[1],[0],[0],[0]],200)
    x=[[0.05],[0.12],[0.15],[0.2],[0.23],[0.67],[0.97],[0.66],[-0.43],[-0.4],[-0.1],[0.9],[0.71],[-0.71],[-0.32]]
    y=[list(map(math.sin,z)) for z in x]
    print(y)
    print(math.sin(0.65))
    network.train(x,y,300)
    print(123)
    print(network.calculate([0.65]))
    print(network.get_weights())