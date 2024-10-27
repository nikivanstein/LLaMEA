import numpy as np

class RefinedMetaheuristicAlgorithm(NovelMetaheuristicAlgorithm):
    def __call__(self, func):
        bounds = (-5.0, 5.0)
        
        # Adjusting the probability of changing individual lines for refinement
        if np.random.rand() < 0.5:
            return self.differential_evolution(func, bounds) + self.particle_swarm_optimization(func, bounds)
        else:
            return self.particle_swarm_optimization(func, bounds) + self.differential_evolution(func, bounds)