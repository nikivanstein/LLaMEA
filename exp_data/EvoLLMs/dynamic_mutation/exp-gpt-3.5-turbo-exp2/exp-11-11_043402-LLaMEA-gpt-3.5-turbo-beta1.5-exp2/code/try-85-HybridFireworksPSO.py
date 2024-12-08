import numpy as np
from scipy.optimize import minimize

class HybridFireworksPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        population_size = 10
        fireworks = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])
        pso_global_best = best_firework
        
        for _ in range(self.budget // population_size - 1):
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            diversity_factor = np.mean(np.std(fireworks, axis=0))  # Dynamic mutation scaling based on population diversity
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= diversity_factor * np.abs(best_firework[j] - fireworks[i][j])
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
            
            pso_local_best = best_firework
            pso_population = fireworks.copy()
            
            def pso_objective(x):
                return func(x)
            
            pso_result = minimize(pso_objective, pso_local_best, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
            pso_global_best = pso_result.x if pso_result.fun < func(pso_global_best) else pso_global_best

        return pso_global_best