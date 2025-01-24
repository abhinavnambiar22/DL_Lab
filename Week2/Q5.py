import torch

def manual(x):

    dax=32*x**3
    dbx=9*x**2
    dcx=14*x
    ddx=6

    dfx=dax+dbx+dcx+ddx

    return dfx

x=torch.tensor(2.0, requires_grad=True)
print('Tensor x:',x)
a=8*x**4
b=3*x**3
c=7*x**2
d=6*x

y=a+b+c+d+3

y.backward()
print('AutoGrad df/dx:',x.grad.item())
print('Manual df/dx:',manual(x).item())

