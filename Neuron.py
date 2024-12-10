from random import random
from sre_constants import error

import math
class Neuron:
    __slots__ = ['weights','value','activation_func','derivative_func']
    derivativeFuncs = {
        "ReLU" : "derivative_ReLU",
        "sigmoid" : "derivative_sigmoid"
    }
    def __init__(self,weights_num:int,activation_func:str):
        self.activation_func=getattr(self,activation_func)
        self.derivative_func=getattr(self,Neuron.derivativeFuncs[activation_func])
        self.weights = []
        self.value:float=0
        for i in range(weights_num):
            self.weights.append(random())

    def ReLU(self,value):
        return max(0,value)

    def derivative_ReLU(self):
        return self.value>0

    def sigmoid(self,value):
        return 1 / (1 + math.exp(-value))

    def derivative_sigmoid(self) -> float:
        #return self.value * (1 - self.value)
        return self.value * (1 - self.value)

    def calculate(self,values:[float]) -> float:
        self.value = self.activation_func(self.__calc_sum(values))
        return self.value

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

