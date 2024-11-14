import numpy as np
from scipy.optimize import minimize

class HybridFireworksPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 10
        fireworks = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        for _ in range(self.budget // population_size - 1):
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            diversity_factor = np.mean(np.std(fireworks, axis=0)
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= diversity_factor * np.abs(best_firework[j] - fireworks[i][j])
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
          
            def objective(x):
                return func(x)
          
            swarm = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
            best_swarm = swarm[np.argmin([func(p) for p in swarm])]
            for _ in range(self.budget // population_size - 1):
                for i in range(population_size):
                    new_particle = swarm[i] + np.random.uniform(0, 1, size=self.dim) * (best_swarm - swarm[i]) + np.random.uniform(0, 1, size=self.dim) * (best_firework - swarm[i])
                    if func(new_particle) < func(swarm[i]):
                        swarm[i] = new_particle
                best_swarm = swarm[np.argmin([func(p) for p in swarm])]
            best_firework = min([best_firework, best_swarm], key=lambda x: func(x))
        
        return best_firework