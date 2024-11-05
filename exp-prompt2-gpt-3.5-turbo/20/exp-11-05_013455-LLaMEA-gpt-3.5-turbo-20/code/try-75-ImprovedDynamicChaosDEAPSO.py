import numpy as np

class ImprovedDynamicChaosDEAPSO(DynamicChaosDEAPSO):
    def chaotic_search(self, x, best, chaos_param):
        new_x = x + chaos_param * np.random.uniform(-5.0, 5.0, x.shape)
        new_x = np.clip(new_x, -5.0, 5.0)
        if func(new_x) < func(x):
            return new_x
        else:
            return x