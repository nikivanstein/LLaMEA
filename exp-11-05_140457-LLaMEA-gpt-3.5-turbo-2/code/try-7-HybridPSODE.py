import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particle_count = 30
        self.max_iter = budget // self.particle_count
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = np.inf

        for _ in range(self.max_iter):
            for _ in range(self.particle_count):
                particle = np.random.uniform(-5.0, 5.0, self.dim)
                velocity = np.random.uniform(-1, 1, self.dim)
                for _ in range(self.budget // (self.max_iter * self.particle_count)):
                    r1, r2 = np.random.uniform(0, 1, 2)
                    
                    mutated_particle = particle + self.F * (best_solution - particle) + self.F * (particle - particle) + self.F * (particle - particle)
                    crossed_particle = []
                    for i in range(self.dim):
                        if np.random.uniform(0, 1) < self.CR or i == np.random.randint(0, self.dim):
                            crossed_particle.append(mutated_particle[i])
                        else:
                            crossed_particle.append(particle[i])
                    particle = np.clip(crossed_particle, -5.0, 5.0)
                    
                    particle_fitness = func(particle)
                    if particle_fitness < best_fitness:
                        best_solution = particle
                        best_fitness = particle_fitness
        return best_solution