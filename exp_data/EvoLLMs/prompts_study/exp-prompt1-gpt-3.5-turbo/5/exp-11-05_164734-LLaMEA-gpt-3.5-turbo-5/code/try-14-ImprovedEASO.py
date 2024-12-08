import numpy as np

class ImprovedEASO(EASO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        def mutate(x, sigma):
            return x + np.random.normal(0, sigma*np.exp(-np.std(x)), len(x))
        
        # Remaining code is unchanged
        <remaining code>