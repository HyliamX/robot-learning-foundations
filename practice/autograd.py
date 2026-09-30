import torch

x = torch.tensor(2.0)
y = torch.tensor(4.0)
w = torch.tensor(3.0, requires_grad=True)
lr = 0.1


for step in range(5):
    pred = w * x
    loss = (pred - y) ** 2
    loss.backward()
    with torch.no_grad():
        w -= w.grad * lr
    print(w.grad)
    w.grad.zero_()
print(w.grad)