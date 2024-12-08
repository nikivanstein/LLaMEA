import numpy as np

class EnhancedChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = [best_individual]  # Elitism to preserve the best individual
            for ind in population[1:]:
                mutation_prob = np.random.rand()
                if mutation_prob < 0.2:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim)  # Exploit best solution
                elif mutation_prob < 0.5:
                    new_ind = best_individual + np.random.uniform(-1.0, 1.0, self.dim)  # Explore around best solution
                else:
                    new_ind = np.random.uniform(-5.0, 5.0, self.dim)  # Diversification
                new_population.append(new_ind)
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]