# import numpy as np

class RefinedMetaheuristicAlgorithm(NovelMetaheuristicAlgorithm):
    def __call__(self, func):
        bounds = (-5.0, 5.0)
        return 0.5 * self.differential_evolution(func, bounds) + 0.5 * self.particle_swarm_optimization(func, bounds)