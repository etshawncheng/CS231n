# code=utf-8
import torch
# • Implement the code for the 2-layer neural networks in CS231n 
# 2021 version with PyTorch (or TensorFlow). 
# • Once you have the code (regardless of which framework you 
# choose above), you will apply your own data.  The training and test 
# dataset is 80%:20%.
# • You need to run the code with the following hyperparameter 
# settings:
# ✓ Activation function: tanh, ReLU;
# ✓ Data preprocessing;
# ✓ Initial weights: small random number, Xavier or Kaiming/MSRA 
# Initialization
# ✓ Loss function: without or with the regularization term (L2), λ = 
# 0.001 or 0.0001
# 𝐸 𝐰 ≡1
# 𝑁

# σ𝑐=1

# 𝑁 𝑓 𝐱𝑐, 𝐰 −𝑦𝑐 2+λ(σ𝑖=0

# 𝑝 (w

# 𝑖

# o)2 + σ𝑖=1

# 𝑝 σ

# 𝑗=0

# 𝑚 (𝑤𝑖𝑗

# 𝐻 )2)

# ✓ Optimizer: gradient descent, Momentum, Adam;
# ✓ Learning epochs: 100, 200, 300;
# ✓ Amount of hidden nodes: 5, 8, 11;
# ✓ Learning rate decay schedule: none and cosine
# ✓ Ensembles: top 3 models
if __name__=="__main__":
    torch.cuda.is_available()