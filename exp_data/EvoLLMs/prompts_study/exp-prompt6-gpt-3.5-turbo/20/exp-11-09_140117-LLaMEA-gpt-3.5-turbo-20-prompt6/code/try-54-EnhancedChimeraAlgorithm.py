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
            new_population = []
            for idx, ind in enumerate(population):
                candidates = [ind for ind in population if not np.array_equal(ind, population[idx])]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = a + 0.8 * (b - c)
                crossover_points = np.random.rand(self.dim) < 0.9
                new_ind = np.where(crossover_points, mutant, ind)
                if func(new_ind) < func(ind):
                    new_population.append(new_ind)
                else:
                    new_population.append(ind)
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]