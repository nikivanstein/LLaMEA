import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim, population_size=50, mutation_factor=0.8, cognitive_weight=0.5, social_weight=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        best_solution = population[np.argmin([func(individual) for individual in population])]

        for _ in range(self.budget):
            for i in range(self.population_size):
                p_best = population[np.argmin([func(individual) for individual in population])]
                velocities[i] = self.cognitive_weight * velocities[i] + self.social_weight * np.random.rand() * (p_best - population[i])
                population[i] = np.clip(population[i] + velocities[i], -5.0, 5.0)

                indices = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                base_vector = population[indices[0]]
                mutated_vector = base_vector + self.mutation_factor * (population[indices[1]] - population[indices[2]])
                trial_vector = np.where(np.random.rand(self.dim) < 0.5, mutated_vector, population[i])

                if func(trial_vector) < func(population[i]):
                    population[i] = trial_vector
                    if func(trial_vector) < func(best_solution):
                        best_solution = trial_vector

        return best_solution