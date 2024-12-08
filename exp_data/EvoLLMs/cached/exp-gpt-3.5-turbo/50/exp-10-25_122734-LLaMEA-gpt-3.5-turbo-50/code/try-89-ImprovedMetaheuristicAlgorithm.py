import numpy as np

class ImprovedMetaheuristicAlgorithm(NovelMetaheuristicAlgorithm):
    def __init__(self, budget, dim):
        NovelMetaheuristicAlgorithm.__init__(self, budget, dim)

    def __call__(self, func):
        bounds = (-5.0, 5.0)
        return self.differential_evolution(func, bounds) + self.particle_swarm_optimization(func, bounds)