import numpy as np

class EnhancedFireworkAlgorithm:
    def __init__(self, budget, dim, n_fireworks=10):
        self.budget = budget
        self.dim = dim
        self.n_fireworks = n_fireworks
        self.n_sparks = 5

    def __call__(self, func):
        def create_firework():
            return np.random.uniform(-5.0, 5.0, self.dim)

        fireworks = [create_firework() for _ in range(self.n_fireworks)]
        best_firework = min(fireworks, key=lambda x: func(x))
        
        for _ in range(self.budget - self.n_fireworks):
            new_fireworks = [firework + np.random.normal(0, 1, self.dim) for firework in fireworks]
            new_sparks = [firework + np.random.normal(0, 0.5 + np.random.random()*0.5, self.dim) for _ in range(self.n_sparks) for firework in fireworks]
            fireworks += new_fireworks + new_sparks
            fireworks.sort(key=lambda x: func(x))
            fireworks = fireworks[:self.n_fireworks]
            if func(fireworks[0]) < func(best_firework):
                best_firework = fireworks[0]
        
        return best_firework