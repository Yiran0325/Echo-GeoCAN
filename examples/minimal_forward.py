import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from geocan import GeoCAN

model = GeoCAN()
x = torch.randn(2, 3, 224, 224)

class_logits, class_pred, score_logits, token_maps = model(x, return_token_maps=True)

print("class_logits:", class_logits.shape)
print("class_pred:", class_pred.shape)
print("score_logits:", score_logits.shape)
print("attention weights psi:", token_maps["psi"].shape)
