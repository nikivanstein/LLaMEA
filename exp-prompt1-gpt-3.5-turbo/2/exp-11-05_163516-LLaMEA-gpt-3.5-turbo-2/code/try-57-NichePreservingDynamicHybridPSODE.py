import numpy as np

class NichePreservingDynamicHybridPSODE(DynamicHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.niche_radius = 0.1

    def __call__(self, func):
        best_position = self.population[np.argmin([func(ind) for ind in self.population])]
        best_fitness = func(best_position)
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                # PSO update
                r1, r2 = np.random.uniform(0, 1, 2)
                new_velocity = self.w * self.population[i] + self.c1 * r1 * (best_position - self.population[i]) + self.c2 * r2 * (best_position - self.population[i])
                new_position = self.population[i] + new_velocity
                new_position = np.clip(new_position, -5.0, 5.0)
                
                # DE update
                rand_indexes = np.random.choice(np.arange(self.pop_size), 3, replace=False)
                mutant = self.population[rand_indexes[0]] + self.f * (self.population[rand_indexes[1]] - self.population[rand_indexes[2]])
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, self.population[i])

                # Niche preservation
                dist_to_others = np.linalg.norm(self.population - self.population[i], axis=1)
                niche_penalty = np.sum(dist_to_others < self.niche_radius) * 0.1
                if func(trial) + niche_penalty < func(self.population[i]):
                    self.population[i] = trial
                if func(new_position) + niche_penalty < func(self.population[i]):
                    self.population[i] = new_position
                
                # Update the global best
                if func(self.population[i]) < best_fitness:
                    best_position = self.population[i]
                    best_fitness = func(best_position)
                    
                    # Dynamic parameter adaptation
                    self.w = max(0.4, self.w * 0.99)
                    self.c1 = max(0.5, self.c1 * 0.99)
                    self.c2 = min(2.0, self.c2 * 1.01)
                    self.cr = min(1.0, self.cr * 1.01)
                    self.f = max(0.5, self.f * 0.99)
        
        return best_position