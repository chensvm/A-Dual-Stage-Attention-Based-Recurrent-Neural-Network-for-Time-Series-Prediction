#! -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input,LSTM,Merge,Add
from keras.models import Model
class My_Dot(Layer):  #参数为(inputs,output_dim)  只改变输入的最后一个维度
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
        super(My_Dot, self).__init__(**kwargs)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',shape=(input_shape[-1],self.output_dim),
                                      initializer='uniform',trainable=True)
        super(My_Dot, self).build(input_shape)

    def call(self, inputs, **kwargs):
        print('Mydot:',K.dot(inputs,self.kernel))
        return K.dot(inputs,self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)

class My_Transpose(Layer):  #参数为(inputs,axis)  改变维度
    def __init__(self,axis,**kwargs):
        self.axis = axis
        super(My_Transpose, self).__init__(**kwargs)
    def build(self, input_shape):
        super(My_Transpose, self).build(input_shape)
    def call(self, inputs, **kwargs):
        return K.permute_dimensions(inputs,pattern=self.axis)
    def compute_output_shape(self, input_shape):
        return (input_shape[self.axis[0]],input_shape[self.axis[1]],input_shape[self.axis[2]])

class My_LSTM(Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
        super(My_LSTM, self).__init__(**kwargs)
    def build(self, input_shape):
        super(My_LSTM, self).build(input_shape)
    def call(self, inputs,**kwargs):
        output_sequence = LSTM(self.output_dim,return_sequences=True)(inputs)
        #output = K.sum(output_sequence,axis=1)
        return output_sequence
    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.output_dim)

a = Input(shape=(10,50))
print(a)
b = My_LSTM(output_dim=5)(a)
model = Model(inputs=a,outputs=b)
print(b)

