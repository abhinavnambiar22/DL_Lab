import torch

def manual_grad(a):

    dxda=2
    dyda=10*a

    dzdx=2
    dzdy=3

    dzda=dzdx*dxda + dzdy*dyda

    return dzda

a=torch.tensor(2.0,requires_grad=True)
b=torch.tensor(3.0,requires_grad=True)

print('Tensor a:',a,'\nTensor b:',b)

x=2*a+3*b
y=5*a**2 + 3*b**3
print('Tensor x:',x,'\nTensor y:',y)

z=2*x+3*y
print('Tensor z:',z)

z.backward()

dzda_manual=manual_grad(a)
print('Gradient dz/da by PyTorch:',a.grad.item())
print('Gradient dz/da manually:',dzda_manual)
