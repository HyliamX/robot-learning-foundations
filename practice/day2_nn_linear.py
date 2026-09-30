import torch
import torch.nn as nn

"""
Day 3: nn.Linear, MSELoss and torch.optim.SGD

Rules:
1. For each question, write your prediction after '# Prediction:' first,
   then write code and run to verify.
2. Record wrong predictions in notes/03_nn_linear.md
   (your prediction, actual result, why).
3. Write the code yourself, do not copy-paste. Check the official PyTorch
   docs when unsure of syntax.
4. Stay within your 2-hour session; skip the challenge if you run out of time.
"""

# ============================================================
# A. nn.Linear basics
# ============================================================

# --- Question 1 ---
# layer = nn.Linear(1, 1). Predict layer.weight.shape and layer.bias.shape,
# then print them.
# Prediction: layer.weight.shape = ([1, 1]), layer.bias.shape = ([1])
layer = nn.Linear(1, 1)
print("layer.weight: ", layer.weight.shape, "\nlayer.bias: ", layer.bias.shape)


# --- Question 2 ---
# x = torch.tensor([[1.0], [2.0], [3.0]])
# Predict the shape of layer(x), then run it.
# Prediction: [3, 1]
x = torch.tensor([[1.0], [2.0], [3.0]])
model = layer(x)
print(model.shape)



# --- Question 3 ---
# nn.Linear(2, 3): predict weight shape and bias shape.
# For an input of shape (5, 2), predict the output shape.
# Prediction:
x = torch.tensor([[1.0 , 6.0], [2., 7.], [3., 8.], [4., 9.], [5., 10.]])
model = nn.Linear(2, 3);
output = model(x)
print(output.shape)


# --- Question 4 ---
# Hand computation: nn.Linear(3, 2). With torch.no_grad(), copy weight rows
# [1,0,0] and [0,1,1] and bias [0.5,-1.0] into the layer (use .copy_).
# For input [[1.,2.,3.]], first compute the two outputs by hand, write them
# as your prediction, then verify by running.
# Also write 1 sentence: why is torch.no_grad() used when setting the
# weights by hand?
# Prediction:
# Why (1-2 sentences):
# TODO


# --- Question 5 ---
# How many learnable numbers (weight + bias) does nn.Linear(576, 10) have?
# Write your prediction, then verify by summing p.numel() over
# layer.parameters().
# Prediction:
# TODO


# --- Question 6 (Concept) ---
# Stack nn.Linear(4, 4) twice with no ReLU between. Write in 1-2 sentences
# why this can never do more than a single linear layer, and what nn.ReLU()
# changes.
# Prediction:
# Why (1-2 sentences):
# TODO


# ============================================================
# B. Loss
# ============================================================

# --- Question 7 ---
# preds = torch.tensor([1., 3.]), targets = torch.tensor([2., 2.])
# Predict nn.MSELoss()(preds, targets) by hand, then verify.
# Prediction:
# TODO


# --- Question 8 ---
# Shape trap: pred has shape (4, 1) and target has shape (4,).
# Predict the shape of pred - target, then run it.
# Write 1 sentence on why this could silently give a wrong loss
# (hint: broadcasting).
# Prediction:
# Why (1-2 sentences):
# TODO


# ============================================================
# C. Optimizer
# ============================================================

# --- Question 9 ---
# p = torch.tensor([3.0], requires_grad=True)
# opt = torch.optim.SGD([p], lr=0.1)
# loss = ((2 * p - 4) ** 2).sum()
# loss.backward()
# Predict p.grad, then predict p after opt.step(), then verify both.
# Prediction:
# TODO


# --- Question 10 ---
# Learning rate experiment: repeat 5 steps of
# (loss, opt.zero_grad(), backward, step) starting from p = 3.0 with lr=0.1
# and again with lr=0.3. Use the same loss as question 9.
# Predict for each lr whether p gets closer to 2 or farther, then print p
# at every step to verify.
# Prediction (lr=0.1):
# Prediction (lr=0.3):
# TODO


# --- Question 11 ---
# In question 10's loop, remove opt.zero_grad(). Predict whether p still
# behaves the same, then run. Write 1 sentence on why.
# Prediction:
# Why (1-2 sentences):
# TODO


# ============================================================
# D. Main exercise: learn y = 2x + 1
# ============================================================

# --- Question 12 ---
# Create x with shape (20, 1) (for example
# torch.linspace(-1, 1, 20).unsqueeze(1), or your own way) and y = 2 * x + 1.
# Predict what the final weight and bias will be close to.
# Prediction:
# TODO


# --- Question 13 ---
# Create model = nn.Linear(1, 1), a loss function (nn.MSELoss), and an
# optimizer (torch.optim.SGD with your choice of lr, start with 0.1).
# Prediction:
# TODO


# --- Question 14 ---
# Write the training loop for 200 steps: forward, loss, zero_grad,
# backward, step. Print the loss every 10 steps.
# Prediction:
# TODO


# --- Question 15 ---
# Print the final model.weight and model.bias and compare with your
# prediction from question 12.
# Prediction:
# TODO


# ============================================================
# E. Challenge (skip if short on time)
# ============================================================

# --- Question 16 ---
# Add noise: y_noisy = y + 0.1 * torch.randn_like(y)
# Predict how the final weight/bias change and whether the loss can
# reach 0. Train again and check.
# Prediction:
# TODO


# --- Question 17 ---
# Try lr = 0.01 and lr = 1.0 on the main exercise.
# Predict what happens to the loss curve in each case, then run.
# Prediction (lr=0.01):
# Prediction (lr=1.0):
# TODO


# ============================================================
# F. Wrap-up (answer in comments, 1-2 sentences each)
# ============================================================

# --- Question 18 ---
# What does opt.step() do that you wrote by hand yesterday?
# Answer:
# TODO


# --- Question 19 ---
# Where is the gradient stored, and who clears it?
# Answer:
# TODO
