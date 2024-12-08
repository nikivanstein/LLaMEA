import numpy as np

class DynamicMutationChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            for ind in population:
                fitness_ratio = func(ind) / func(best_individual)
                mutation_prob = np.random.rand()
                if mutation_prob < 0.3:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim) * fitness_ratio
                elif mutation_prob < 0.6:
                    new_ind = ind + np.random.randn(self.dim) * fitness_ratio
                else:
                    new_ind = np.random.uniform(-5.0, 5.0, self.dim)
                new_population.append(new_ind)
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]