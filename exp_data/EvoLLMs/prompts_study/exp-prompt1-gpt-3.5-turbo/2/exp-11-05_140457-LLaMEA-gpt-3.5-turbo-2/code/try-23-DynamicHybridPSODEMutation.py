import numpy as np

class DynamicHybridPSODEMutation(DynamicHybridPSODE):
    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = np.inf
        mutation_step = 0.1

        for _ in range(self.max_iter):
            for _ in range(self.particle_count):
                particle = np.random.uniform(-5.0, 5.0, self.dim)
                velocity = np.random.uniform(-1, 1, self.dim)
                for _ in range(self.budget // (self.max_iter * self.particle_count)):
                    r1, r2 = np.random.uniform(0, 1, 2)
                    
                    self.CR = max(0.1, min(0.9, self.CR + np.random.normal(0, 0.1)))
                    
                    mutated_particle = particle + mutation_step * (best_solution - particle) + mutation_step * (particle - particle) + mutation_step * (particle - particle)
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
                        mutation_step = max(0.01, min(0.5, mutation_step * 1.2)) if particle_fitness <= best_fitness else max(0.01, min(0.5, mutation_step * 0.8))
        return best_solution