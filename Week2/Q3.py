import torch

def analytical(x,v):

    #Sigmoid Gradient starts
    x1=-v
    x2=torch.exp(x1)
    x3=1+x2
    a=1/x3

    dax3=-1/(x3**2)
    dx3x2=1
    dx2x1=x2
    dx1dv=-1

    dadv=dax3*dx3x2*dx2x1*dx1dv
    #Sigmoid Gradient Ends

    dvdu=1
    dudw=x
    dadw=dadv*dvdu*dudw

    return dadw

x=torch.tensor(2.0)
w=torch.tensor(3.0,requires_grad=True)
b=torch.tensor(5.0)

print('Input:',x,'\nWeight:',w,'\nBias:',b)

v=w*x+b
print('Weighted Sum:',v)

a=1/(1+torch.exp(-v))

a.backward()

print('Autograd da/dw:',w.grad.item())
print('Manual da/dw: ',analytical(x,v))


