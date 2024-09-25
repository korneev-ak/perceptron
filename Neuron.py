from random import random
from sre_constants import error

import math
class Neuron:
    __slots__ = ['weights','value']
    def __init__(self,weights_num:int):
        self.weights = []
        self.value:int=0
        for i in range(weights_num):
            self.weights.append(random())

    def activation(self,value):
        #self.value=1 / (1 + math.exp(-value))
        self.value=max(value,0)
        return self.value



    def derivative_sigmoid(self) -> float:
        #return self.value * (1 - self.value)
        return self.value>0

    def calculate(self,values:[float]) -> float:
        return self.activation(self.__calc_sum(values))

    def __calc_sum(self,values:[float]) -> float:
        if len(values) != len(self.weights):
            raise Exception('different number of input values and weights number')
        return sum([weight * value for weight,value in zip(self.weights,values)])

    def get_weights(self):
        return self.weights

    def set_weights(self,weights):
        for i in range(len(weights)):
            self.weights[i]=weights[i]

    def adjust_weights(self,my_error:float,input_data):
        for i in range(len(input_data)):
            self.weights[i] += my_error*0.33*input_data[i]

