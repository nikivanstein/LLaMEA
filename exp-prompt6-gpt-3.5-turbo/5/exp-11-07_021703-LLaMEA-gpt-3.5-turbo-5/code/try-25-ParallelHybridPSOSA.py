import numpy as np
import random
from multiprocessing import Pool

class ParallelHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.max_iter = budget // self.num_particles

    def optimize_particles(self, particles, func):
        # PSO and SA code here
        return particles

    def __call__(self, func):
        # Initialization
        best_solution = np.random.uniform(-5.0, 5.0, [self.dim])
        best_fitness = func(best_solution)

        # Main optimization loop
        for _ in range(self.max_iter):
            particles = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
            
            # Parallel processing for particle evaluation
            with Pool() as p:
                particles = p.starmap(self.optimize_particles, [(particle, func) for particle in particles])
            
            fitness = func(particles.T)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = particles[best_idx]
                best_fitness = fitness[best_idx]

        return best_solution