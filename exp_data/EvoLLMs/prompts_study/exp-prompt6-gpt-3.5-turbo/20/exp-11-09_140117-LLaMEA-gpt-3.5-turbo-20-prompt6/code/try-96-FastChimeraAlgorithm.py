import numpy as np

class FastChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_probs = [0.3, 0.6, 0.9]
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            for ind in population:
                mutation_prob = np.random.choice(self.mutation_probs)
                if mutation_prob < 0.4:
                    new_ind = ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim)
                elif mutation_prob < 0.7:
                    new_ind = ind + np.random.randn(self.dim)
                else:
                    new_ind = np.random.uniform(-5.0, 5.0, self.dim)
                new_population.append(new_ind)
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]