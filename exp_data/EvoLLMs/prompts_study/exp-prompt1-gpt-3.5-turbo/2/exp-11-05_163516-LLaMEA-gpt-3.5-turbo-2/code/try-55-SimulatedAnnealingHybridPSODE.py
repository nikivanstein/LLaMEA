import numpy as np

class SimulatedAnnealingHybridPSODE(DynamicHybridPSODE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.initial_temperature = 10.0
        self.final_temperature = 0.1
        
    def __call__(self, func):
        best_position = self.population[np.argmin([func(ind) for ind in self.population])]
        best_fitness = func(best_position)
        
        temperature = self.initial_temperature
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                # Simulated Annealing inspired mutation
                new_position = self.population[i] + np.random.normal(0, 1, self.dim) * temperature
                new_position = np.clip(new_position, -5.0, 5.0)
                
                if func(new_position) < func(self.population[i]):
                    self.population[i] = new_position
                elif np.random.rand() < np.exp((func(self.population[i]) - func(new_position)) / temperature):
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
                
            temperature = max(self.final_temperature, temperature * 0.9)
        
        return best_position