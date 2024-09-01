# https://www.youtube.com/watch?v=EMXfZB8FVUA&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4

# you will run this in conda env in VS code, to get intellisense, press Ctrl Shift P,
# Python: Select Interpreter and select conda env specific python interpreter
import torch

print("Chapter 2 Tensors and operations") 
print("\n1d matrix") 
x= torch.empty(1)
print(x)
print("\n2d matrix") 
x= torch.empty(2,2)
print(x)
print("\n3d matrix") 
x= torch.empty(2,2,3)
print(x)
print("\nrandom init")
x=torch.rand(2,2)
print(x)
print("\nzero set tensor")
x=torch.zeros(2,2)
print(x)
print("one set tensor")
x=torch.ones(2,2)
print(x)

print("\ndatatype")
x=torch.ones(2,2)
print(x.dtype)

print("\ndatatype int set")
x=torch.ones(2,2,dtype=torch.int)
print(x.dtype)
print("\ndatatype double (float 64) set")
x=torch.ones(2,2,dtype=torch.double)
print(x.dtype)
print("\ndatatype  (float 16) set")
x=torch.ones(2,2,dtype=torch.float16)
print(x.dtype)

print("\nsize")
x=torch.ones(2,2,dtype=torch.float16)
print(x.size())

print("\nconstuct tensor from data (python list)")
x=torch.tensor([2.5,0.1])
print(x)

print("\ncreate 2 tensors with random values and add them")
x=torch.rand(2,2)
y=torch.rand(2,2)
print(x)
print(y)
z=x+y
print(z)
z=torch.add(x,y) #same as above
print(z)

print("\ncreate 2 tensors with random values and subtract one from another")
x=torch.rand(2,2)
y=torch.rand(2,2)
print(x)
print(y)
z=x-y
print(z)
z=torch.sub(x,y) #same as above
print(z)

print("\ncreate 2 tensors with random values and multiply")
x=torch.rand(2,2)
y=torch.rand(2,2)
print(x)
print(y)
z=x*y
print(z)
z=torch.mul(x,y) #same as above
print(z)

print("\ncreate 2 tensors with random values and divide")
x=torch.rand(2,2)
y=torch.rand(2,2)
print(x)
print(y)
z=x*y
print(z)
z=torch.div(x,y) #same as above
print(z)

print("\n Inplace addition using underscore functions (add and assign to the same tensor)")
x=torch.rand(2,2)
y=torch.rand(2,2)
y.add_(x)
print(y)

print("\n Inplace subtract using underscore functions (subtract and assign to the same tensor)")
x=torch.rand(2,2)
y=torch.rand(2,2)
y.sub_(x)
print(y)

print("\n Inplace multiplication using underscore functions (multiply and assign to the same tensor)")
x=torch.rand(2,2)
y=torch.rand(2,2)
y.mul_(x)
print(y)

print("\n Inplace division using underscore functions (divide and assign to the same tensor)")
x=torch.rand(2,2)
y=torch.rand(2,2)
y.div_(x)
print(y)

print("\n slicing of tensor")
x=torch.rand(5,3)
print(x)
print(f"This operation is a slicing operation - get all rows but one column 0 only:\n {x[:, 0]}")
print(f"This operation is a slicing operation - get only row 1 but all columns:\n {x[1, :]}")
print(f"This operation is a slicing operation - get only row 1 and column 1 element:\n {x[1, 1]}")

print(f"This operation is a slicing operation - get only row 1 and column 1 element:\n {x[1, 1].item()}")
#be careful in above, if there are multiple values, it will fail


print("\n Reshape tensor")
x=torch.rand(4,4)
print(x)
y=x.view(16)
print(f"\n{y}")

x=torch.rand(4,4)
print(x)
print(f"If you need pytorch to figure out the other dimension automatically, then put -1. In this case it will do 2x8 tensor ")
y=x.view(-1,8)
print(f"\n{y}")
print(f"\n{y.size()}")


print("\n Convert a torch tensor to numpy array")
import numpy as np
a=torch.ones(5)
print(a)
b=a.numpy()
print(f"\nConverted to numpy array: {b} ")
print(f"\nConverted to numpy array check type: {type(b)} ")

print("\nIf CPU is used, changing or covnerting one of the tensor will also change the another. ")
print("For example, if you add 1 to a tensor then b array will automatically add 1 too when CPU is used")
a.add_(1)
print(f"\ntorch tensor: {a}")
print(f"numpy array: {b}")

print("\n Convert a numpy array to torch tensor")
import numpy as np
a=np.ones(5)
print(a)
b=torch.from_numpy(a)
print(f"\nConverted to torch sensor: {b} ")
print(f"\nConverted to torch sensor check type: {type(b)} ")
print(f"\nYou can specifiy the datatype to the converted to post conversion")
b=torch.from_numpy(a)
#commenting out below to show the CPU / GPU difference
# b = b.to(torch.int)  # or b = b.int()
# print(f"\nConverted to torch sensor with datatype int: {b} ")
# print(f"\nConverted to torch sensor check type: {type(b)} ")

print("\nIf CPU is used, changing or covnerting one of the tensor will also change the another. ")
print("For example, if you add 1 to a tensor then b tensor will automatically add 1 too when CPU is used")
a+= 1
print(f"\nnumpy array: {a}")
print(f"torch tensor: {b}")

print("\ndo operation on GPU ")
if torch.cuda.is_available():
    device=torch.device("cuda")
    # creates tensor on GPU
    x=torch.ones(5,device=device) 
    # creates tensor on CPU and moves to GPU
    y=torch.ones(5)
    y=y.to(device)
    
    z = x + y # addition will take place in GPU
    print(z)
    #foll will now caause error as numpy can only handle CPU tensors/array and not GPU tensors/array
    #z.numpy()
    #to convert it to numpy you would have to move it back to cpu
    z = z.to("cpu")
    z.numpy()
    print(f"since z is now in CPU, conversion to numpy will work: {z}")









