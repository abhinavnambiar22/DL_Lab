import torch

def manual(x,d):

    dfdd=torch.exp(d)

    ddda=1
    dddb=1
    dddc=1

    dadx=-2*x
    dbdx=-2
    dcdx=-torch.cos(x)

    dfdx=dfdd*(ddda*dadx + dddb*dbdx + dddc*dcdx)

    return dfdx


x=torch.tensor(2.0, requires_grad=True)
print('Tensor x:',x)
a=-x**2
b=-2*x
c=-torch.sin(x)
d=a+b+c

f=torch.exp(d)
print('Tensor f:',f)

f.backward()
print('AutoGrad df/dx:',x.grad.item())
print('Manual df/dx:',manual(x, d).item())