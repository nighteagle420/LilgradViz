from engine import *

a = TensorVal([1.0,2.0,3.0])
b = TensorVal([1,2,3,4])

c = a@b
print(c)