import numpy as np

class EnhancedChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.3
        self.mutation_scale = 0.1

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            for ind in population:
                mutation_prob = np.random.rand()
                if mutation_prob < self.mutation_rate:
                    mutation_step = self.mutation_scale * np.linalg.norm(best_individual - ind)
                    new_ind = ind + mutation_step * np.random.uniform(0.0, 1.0, self.dim)
                elif mutation_prob < 2 * self.mutation_rate:
                    new_ind = ind + np.random.randn(self.dim)
                else:
                    new_ind = np.random.uniform(-5.0, 5.0, self.dim)
                new_population.append(new_ind)
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]