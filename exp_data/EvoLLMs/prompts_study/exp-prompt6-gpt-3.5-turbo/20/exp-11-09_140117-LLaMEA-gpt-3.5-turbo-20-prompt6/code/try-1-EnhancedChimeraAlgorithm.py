import numpy as np

class EnhancedChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        mutation_probs = np.full(self.budget, 0.5)
        
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            
            for idx, ind in enumerate(population):
                if np.random.rand() < mutation_probs[idx]:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim)
                else:
                    new_ind = ind + np.random.randn(self.dim)
                
                new_population.append(new_ind)
                
                if func(new_ind) < func(ind):
                    mutation_probs[idx] += 0.1
                else:
                    mutation_probs[idx] -= 0.1
                    
                mutation_probs[idx] = np.clip(mutation_probs[idx], 0.1, 0.9)
                    
            population = np.array(new_population)
            
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]