import torch

x = torch.ones(5)  # input tensor
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b

y = torch.zeros(3)  # expected output
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(z)
print(y)
print(loss)

z.grad_fn
loss.grad_fn

# Computing gradients

loss.backward()
print(w.grad)
print(b.grad)


# Disable gradient tracking

z = torch.matmul(x, w)+b
print(z.requires_grad)


with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)


z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

