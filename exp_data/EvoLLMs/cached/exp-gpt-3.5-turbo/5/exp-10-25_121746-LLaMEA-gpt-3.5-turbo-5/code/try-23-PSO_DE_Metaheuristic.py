import numpy as np

class PSO_DE_Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.cr = 0.5
        self.f = 0.5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

        def mutate(x, r1, r2, r3):
            return x + self.f * (r1 - x) + self.f * (r2 - x)

        def crossover(x, trial, dim_to_cross):
            crossed = [trial[i] if np.random.rand() < self.cr or i == dim_to_cross else x[i] for i in range(self.dim)]
            return np.clip(crossed, self.lb, self.ub)

        population = initialize_population()
        fitness_values = [objective_function(ind) for ind in population]
        best_index = np.argmin(fitness_values)
        best_individual = population[best_index]

        for _ in range(self.budget - self.pop_size):
            new_population = []
            for i in range(self.pop_size):
                r1, r2, r3 = population[np.random.choice(self.pop_size, 3, replace=False)]
                trial_individual = mutate(population[i], r1, r2, r3)
                dim_to_cross = np.random.randint(0, self.dim)
                trial_individual = crossover(population[i], trial_individual, dim_to_cross)
                new_population.append(trial_individual)

            new_fitness_values = [objective_function(ind) for ind in new_population]
            for i in range(self.pop_size):
                if new_fitness_values[i] < fitness_values[i]:
                    population[i] = new_population[i]
                    fitness_values[i] = new_fitness_values[i]
                    if new_fitness_values[i] < fitness_values[best_index]:
                        best_index = i
                        best_individual = population[i]

        return objective_function(best_individual)
