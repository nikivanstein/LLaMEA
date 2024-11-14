import numpy as np

class ImprovedChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]
            new_population = []
            for _ in range(self.budget):
                parent1 = population[np.random.randint(0, self.budget)]
                parent2 = population[np.random.randint(0, self.budget)]
                crossover_prob = np.random.rand()
                if crossover_prob < 0.5:
                    crossover_point = np.random.randint(0, self.dim)
                    new_ind = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                else:
                    new_ind = parent1
                mutation_prob = np.random.rand()
                if mutation_prob < 0.3:
                    new_ind = new_ind + (best_individual - new_ind) * np.random.uniform(0.0, 1.0, self.dim)
                elif mutation_prob < 0.6:
                    new_ind = new_ind + np.random.randn(self.dim)
                else:
                    new_ind = np.clip(new_ind, -5.0, 5.0)
                new_population.append(new_ind)
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]