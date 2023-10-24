import torch
from gymnasium import spaces
import numpy as np
test = spaces.Box(low=0,high=100,shape=(2,), dtype=int)
test2 = spaces.MultiDiscrete([4, 4, 4, 4, 4])
t = torch.Tensor([2, 2, 2, 2, 2])
# print(test2.contains(t.numpy()))
point = np.random.random_integers(low=0,high=10,size=(2,))
print(point)