import numpy as np

class EnhancedHDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.cr = 0.9
        self.f = 0.8
        self.alpha = 0.9
        self.sigma = 0.1
        self.history_weight = 0.1  # New parameter

    def __call__(self, func):
        def de_mutate(population, target_idx):
            candidates = population[np.random.choice(np.delete(np.arange(self.pop_size), target_idx), 3, replace=False)]
            donor_vector = population[target_idx] + self.f * (candidates[0] - candidates[1])
            for i in range(self.dim):
                if np.random.rand() > self.cr:
                    donor_vector[i] = population[target_idx][i]
            return donor_vector

        def sa_mutation(candidate, best, t):
            return candidate + self.sigma * np.exp(-t * self.alpha) * np.random.normal(0, 1, self.dim)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_solution_history = np.copy(best_solution)  # New variable
        t = 0

        while t < self.budget:
            new_population = np.zeros_like(population)
            for i in range(self.pop_size):
                candidate = de_mutate(population, i)
                candidate_fitness = func(candidate)
                if candidate_fitness < fitness[i]:
                    new_population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < fitness[best_idx]:
                        best_solution = candidate
                        best_idx = i
                else:
                    history_mutation = self.history_weight * best_solution_history  # New line
                    new_population[i] = sa_mutation(population[i] + history_mutation, best_solution, t)  # Updated line
                t += 1

            population = new_population

        return best_solution