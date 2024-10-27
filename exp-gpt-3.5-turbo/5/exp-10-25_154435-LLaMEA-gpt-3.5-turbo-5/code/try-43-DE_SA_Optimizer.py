import numpy as np

class DE_SA_Optimizer:
    def __init__(self, budget, dim, population_size=50, mutation_factor=0.8, crossover_prob=0.9, cooling_rate=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.cooling_rate = cooling_rate

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])]
        for _ in range(self.budget):
            target_vectors = []
            for i in range(self.population_size):
                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                base_vector = population[indices[0]]
                mutated_vector = base_vector + self.mutation_factor * (population[indices[1]] - population[indices[2]])
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_prob, mutated_vector, population[i])
                
                current_cost = func(population[i])
                trial_cost = func(trial_vector)
                if trial_cost < current_cost:
                    population[i] = trial_vector
                    if trial_cost < func(best_solution):
                        best_solution = trial_vector
                else:
                    if np.random.rand() < np.exp((current_cost - trial_cost) / self.cooling_rate):
                        population[i] = trial_vector

        return best_solution