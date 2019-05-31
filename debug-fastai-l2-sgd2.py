from fastai.basics import *

n=100


x = torch.ones(n,2) 
x[:,0].uniform_(-1.,1)
x[:5]


a = tensor(3.,2); a

y = x@a + torch.rand(n)

def mymse(y_hat, y): 
    return ((y_hat-y)**2).mean()


a = tensor(-1.,1)

a = tensor(3.0,2.5)

y_hat = x@a
mse(y_hat, y)

a = tensor(-1.,1)


a = nn.Parameter(a); a


def update():
    y_hat = x@a
    loss = mymse(y, y_hat)
    if t % 10 == 0: print(loss)
    loss.backward()
    with torch.no_grad():
        a.sub_(lr * a.grad)
        a.grad.zero_()

lr = 1e-1
for t in range(100): update()




