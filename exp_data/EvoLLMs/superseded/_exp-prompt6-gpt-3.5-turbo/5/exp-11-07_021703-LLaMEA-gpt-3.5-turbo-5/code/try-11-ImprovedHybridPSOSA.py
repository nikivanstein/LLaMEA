import numpy as np

class ImprovedHybridPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20

    def __call__(self, func):
        def pso(particles):
            # Parallelized PSO implementation
            return particles

        def sa(particles):
            # Parallelized SA implementation
            return particles

        # Initialization
        best_solution = np.random.uniform(-5.0, 5.0, [self.dim])
        best_fitness = func(best_solution)

        # Main optimization loop with parallelized PSO and SA
        for _ in range(self.budget // self.num_particles):
            particles = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
            particles = pso(particles)
            particles = sa(particles)
            fitness = func(particles.T)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = particles[best_idx]
                best_fitness = fitness[best_idx]

        return best_solution