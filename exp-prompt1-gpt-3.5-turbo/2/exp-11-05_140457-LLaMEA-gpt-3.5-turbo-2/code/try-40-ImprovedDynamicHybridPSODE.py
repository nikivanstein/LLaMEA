import numpy as np

class ImprovedDynamicHybridPSODE(DynamicHybridPSODE):
    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = np.inf

        for _ in range(self.max_iter):
            for _ in range(self.particle_count):
                particle = np.random.uniform(-5.0, 5.0, self.dim)
                velocity = np.random.uniform(-1, 1, self.dim)
                for _ in range(self.budget // (self.max_iter * self.particle_count)):
                    r1, r2 = np.random.uniform(0, 1, 2)

                    self.F = max(0.1, min(0.9, self.F + np.random.normal(0, 0.1)))
                    self.CR = max(0.1, min(0.9, self.CR + np.random.normal(0, 0.1)))

                    mutated_particle = particle + self.F * (best_solution - particle) + self.F * (particle - particle) + self.F * (particle - particle)
                    crossed_particle = []
                    for i in range(self.dim):
                        if np.random.uniform(0, 1) < self.CR or i == np.random.randint(0, self.dim):
                            crossed_particle.append(mutated_particle[i])
                        else:
                            crossed_particle.append(particle[i])
                    particle = np.clip(crossed_particle, -5.0, 5.0)
                    
                    # New mutation mechanism based on adaptive differential evolution
                    for i in range(self.dim):
                        if np.random.uniform(0, 1) < 0.5 or i == np.random.randint(0, self.dim):
                            particle[i] = particle[i] + np.random.uniform(-0.5, 0.5) * np.linalg.norm(particle - best_solution)
                    
                    particle_fitness = func(particle)
                    if particle_fitness < best_fitness:
                        best_solution = particle
                        best_fitness = particle_fitness
                        
                    self.F = max(0.1, min(0.9, self.F + np.random.normal(0, 0.1) * (1 - particle_fitness / best_fitness)))
        return best_solution