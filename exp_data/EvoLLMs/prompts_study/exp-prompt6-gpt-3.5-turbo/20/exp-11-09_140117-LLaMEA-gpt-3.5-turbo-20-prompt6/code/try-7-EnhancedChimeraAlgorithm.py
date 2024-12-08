import numpy as np

class EnhancedChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        mutation_rates = np.full(self.budget, 0.3)
        
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            
            for idx, ind in enumerate(population):
                if idx == best_idx:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim)
                else:
                    new_ind = ind + np.random.randn(self.dim) * mutation_rates[idx]
                    
                new_population.append(new_ind)
                mutation_rates[idx] *= 0.99  # Reduce mutation rate gradually
                
            population = np.array(new_population)
            
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]