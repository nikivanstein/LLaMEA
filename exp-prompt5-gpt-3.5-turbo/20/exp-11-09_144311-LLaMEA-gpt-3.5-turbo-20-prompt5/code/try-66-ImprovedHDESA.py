import numpy as np

class ImprovedHDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.min_pop_size = 6
        self.max_pop_size = 12
        self.cr = 0.9
        self.f = 0.8
        self.alpha = 0.9
        self.sigma = 0.1
        self.adaptive_param = 0.1  # Adaptive control parameter
        self.learning_rate = 0.05  # Dynamic learning rate

    def __call__(self, func):
        def de_mutate(population, target_idx):
            candidates = population[np.random.choice(np.delete(np.arange(len(population)), target_idx), 3, replace=False)]
            self.f = max(0.1, min(0.9, self.f + np.random.normal(0, self.adaptive_param)))  # Adaptive control
            self.learning_rate = max(0.01, min(0.1, self.learning_rate + np.random.normal(0, self.adaptive_param)))  # Dynamic learning rate adjustment
            donor_vector = population[target_idx] + (self.f + self.learning_rate) * (candidates[0] - candidates[1])
            for i in range(self.dim):
                if np.random.rand() > self.cr:
                    donor_vector[i] = population[target_idx][i]
            return donor_vector

        def sa_mutation(candidate, best, t):
            self.sigma = max(0.01, min(0.5, self.sigma + np.random.normal(0, self.adaptive_param)))  # Adaptive control
            self.learning_rate = max(0.01, min(0.1, self.learning_rate + np.random.normal(0, self.adaptive_param)))  # Dynamic learning rate adjustment
            return candidate + (self.sigma + self.learning_rate) * np.exp(-t * self.alpha) * np.random.normal(0, 1, self.dim)

        population = np.random.uniform(-5.0, 5.0, (self.min_pop_size, self.dim))  # Start with minimum population size
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        t = 0

        while t < self.budget:
            new_population = np.zeros((len(population), self.dim))  # Initialize new population based on dynamic size
            for i in range(len(population)):
                candidate = de_mutate(population, i)
                candidate_fitness = func(candidate)
                if candidate_fitness < fitness[i]:
                    new_population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < fitness[best_idx]:
                        best_solution = candidate
                        best_idx = i
                else:
                    new_population[i] = sa_mutation(population[i], best_solution, t)
                t += 1

            if np.std(fitness) > 0.1:  # Adjust population size based on fitness diversity
                population = np.vstack((population, np.random.uniform(-5.0, 5.0, (1, self.dim))))
                fitness = np.append(fitness, func(population[-1]))

            if len(population) > self.max_pop_size:  # Maintain population size within bounds
                worst_idx = np.argmax(fitness)
                population = np.delete(population, worst_idx, axis=0)
                fitness = np.delete(fitness, worst_idx)

        return best_solution