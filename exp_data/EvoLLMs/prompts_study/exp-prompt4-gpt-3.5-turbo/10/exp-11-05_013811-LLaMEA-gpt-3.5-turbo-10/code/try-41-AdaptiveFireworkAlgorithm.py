import numpy as np

class AdaptiveFireworkAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.n_fireworks = 10
        self.n_sparks = 5
        self.mutation_scale = 1.0

    def __call__(self, func):
        def create_firework():
            return np.random.uniform(-5.0, 5.0, self.dim)

        fireworks = [create_firework() for _ in range(self.n_fireworks)]
        best_firework = min(fireworks, key=lambda x: func(x))
        
        for _ in range(self.budget - self.n_fireworks):
            new_fireworks = [firework + np.random.standard_cauchy(self.dim) for firework in fireworks]  # Cauchy mutation for increased exploration
            new_sparks = [firework + np.random.standard_cauchy(self.dim) * 0.2 for _ in range(self.n_sparks) for firework in fireworks]  # Increased sparks diversity with Cauchy mutation
            fireworks += new_fireworks + new_sparks
            fireworks.sort(key=lambda x: func(x))
            fireworks = fireworks[:self.n_fireworks]
            if func(fireworks[0]) < func(best_firework):
                best_firework = fireworks[0]
            self.mutation_scale = max(0.1, self.mutation_scale * 0.99)  # Adaptive mutation strategy
            
            # Dynamic population size strategy to adaptively adjust the number of fireworks
            if np.random.rand() < 0.1:
                self.n_fireworks = max(5, int(self.n_fireworks * 1.1)) if np.random.rand() < 0.5 else max(5, int(self.n_fireworks * 0.9))

        return best_firework