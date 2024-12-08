import numpy as np

class ImprovedDynamicHybridPSODE(DynamicHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 10
        self.min_population_size = 5
        self.max_population_size = 20

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = np.inf

        for _ in range(self.max_iter):
            for _ in range(self.population_size):
                particle = np.random.uniform(-5.0, 5.0, self.dim)
                velocity = np.random.uniform(-1, 1, self.dim)
                for _ in range(self.budget // (self.max_iter * self.population_size)):
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
                    
                    particle_fitness = func(particle)
                    if particle_fitness < best_fitness:
                        best_solution = particle
                        best_fitness = particle_fitness

            # Dynamic population size adaptation
            improvement_ratio = (best_fitness - self.F_prev_best) / (self.F_prev_best + 1e-8)
            if improvement_ratio > 0.1 and self.population_size < self.max_population_size:
                self.population_size += 1
            elif improvement_ratio < 0.01 and self.population_size > self.min_population_size:
                self.population_size -= 1

            self.F_prev_best = best_fitness

        return best_solution