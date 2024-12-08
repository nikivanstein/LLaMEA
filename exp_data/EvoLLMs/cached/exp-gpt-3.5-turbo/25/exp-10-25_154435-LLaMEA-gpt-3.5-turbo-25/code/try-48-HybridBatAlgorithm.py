import numpy as np

class HybridBatAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10
        self.frequency_min = 0.0
        self.frequency_max = 2.0
        self.loudness = 1.0
        self.pulse_rate = 0.5
        
    def __call__(self, func):
        population = [np.random.uniform(self.lower_bound, self.upper_bound, self.dim) for _ in range(self.population_size)]
        velocities = [np.zeros(self.dim) for _ in range(self.population_size)]
        best_solution = population[np.argmin([func(ind) for ind in population])]
        
        for _ in range(self.budget):
            for i, bat in enumerate(population):
                frequency = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand()
                velocities[i] += (bat - best_solution) * frequency
                new_bat = bat + velocities[i]
                
                if np.random.rand() > self.pulse_rate:
                    new_bat = best_solution + 0.001 * np.random.normal(0, 1, self.dim)
                
                if func(new_bat) < func(bat) and np.random.rand() < self.loudness:
                    population[i] = new_bat
                
                best_solution = population[np.argmin([func(ind) for ind in population])]
        
        return best_solution