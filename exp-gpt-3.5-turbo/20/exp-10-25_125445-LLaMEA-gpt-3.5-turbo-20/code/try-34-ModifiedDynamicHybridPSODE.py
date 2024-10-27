import numpy as np

class ModifiedDynamicHybridPSODE(DynamicHybridPSODE):
    def __init__(self, budget, dim, population_size=30, w=0.5, c1=1.5, c2=1.5, f=0.5, cr=0.9, min_w=0.4, max_w=0.9, min_c=0.5, max_c=2.0, min_f=0.3, max_f=0.8, min_cr=0.5, max_cr=1.0):
        super().__init__(budget, dim, population_size, w, c1, c2, f, cr, min_w, max_w, min_c, max_c, min_f, max_f, min_cr, max_cr)

    def adapt_parameters(self, func, gbest_val):
        improvement_threshold = 0.1
        if np.random.rand() < 0.2:
            if gbest_val < improvement_threshold:
                for param in ['w', 'c1', 'c2', 'f', 'cr']:
                    setattr(self, param, np.clip(getattr(self, param) * 1.1, getattr(self, 'min_' + param), getattr(self, 'max_' + param)))