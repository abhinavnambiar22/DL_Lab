import torch

x=torch.tensor(2.0)
y=torch.tensor(3.0,requires_grad=True)
z=torch.tensor(4.0)
print('Tensor x:',x,'\nTensor y:',y,'\nTensor z:',z)
a=2*x
b=torch.sin(y)
c=a/b
d=c*z
e=torch.log(d+1)
f=torch.tanh(e)

print('Intermediate variables:',a,b,c,d,e,f)

#Manual Gradient Starts
dfe=1-(torch.tanh(e))**2
ded=1/(d+1)
ddc=z
dcb=-a/b**2
dby=torch.cos(y)

manual_grad=dfe*ded*ddc*dcb*dby

f.backward()
print('AutoGrad df/dx:',y.grad.item())
print('Manual df/dx:',manual_grad.item())
