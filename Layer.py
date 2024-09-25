from dbm.dumb import error

from Neuron import Neuron
class Layer:
    __slots__ = ['neurons','next_layer']
    def __init__(self,layers:[int],weights_num:int):
        if isinstance(layers,int):
            layers=[layers]

        self.neurons=[Neuron(weights_num) for _ in range(layers[0])]

        if len(layers)==1:
            self.next_layer=None #if it's output layer
        else:
            self.next_layer=Layer(layers[1:],len(self.neurons))

    def train(self,data:[float],y:[float]) -> ([float],[[float]]):
        result=[neuron.calculate(data) for neuron in self.neurons] #must be calculated anyway to give values to neurons
        returning_weights: [[float]] = [] #for returning back to previous layer

        for i in range(len(self.neurons[0].weights)):
            returning_weights.append(([neuron.weights[i] for neuron in self.neurons]))

        if self.next_layer==None:
            local_error:[float] = [(answer-neuron.value)*neuron.derivative_sigmoid() for neuron,answer in zip(self.neurons,y)]
            [neuron.adjust_weights(err,data) for neuron,err in zip(self.neurons,local_error)]
            return local_error,returning_weights

        returned_error,returned_weight=self.next_layer.train(result,y)
        local_error=self.calc_error(returned_error,returned_weight)
        """
        local_error:[float]=[]
        for i in range(len(returned_error)):
            s=sum([returned_error[i]*returned_weight[i][k] for k in range(len(returned_weight[i]))])
            local_error+=[s*self.neurons[i].derivative_sigmoid()]
        """

        [neuron.adjust_weights(err, data) for neuron, err in zip(self.neurons, local_error)]
        return local_error,returning_weights

    def calc_error(self,err:[float],weights:[float]) -> [float]:
        result=[]
        for i in range(len(err)):
            s=sum([err[i]*weights[i][k] for k in range(len(weights[i]))])
            result+=[s*self.neurons[i].derivative_sigmoid()]
        return result


    def get_weights(self):
        if self.next_layer==None:
            return [neuron.get_weights() for neuron in self.neurons]
        else:
            return [neuron.get_weights() for neuron in self.neurons] + self.next_layer.get_weights()

    def forward(self,values:[float]) -> [float]:
        result = [neuron.calculate(values) for neuron in self.neurons]
        if self.next_layer==None:
            return result
        return self.next_layer.forward(result)

    def set_weights(self,weights:[[[float]]]):
        if len(weights)>0:
            [neuron.set_weights(w) for neuron,w in zip(self.neurons,weights[0])]
            if self.next_layer!=None:
                self.next_layer.set_weights(weights[1:])

