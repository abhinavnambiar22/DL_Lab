import torch

def analytical(x,v):

    if v<0:
        dadv=0
    else:
        dadv=1

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

if v>0:
    a=v
else:
    a=0

a.backward()

print('Autograd da/dw:',w.grad.item())
print('Manual da/dw: ',analytical(x,v))


