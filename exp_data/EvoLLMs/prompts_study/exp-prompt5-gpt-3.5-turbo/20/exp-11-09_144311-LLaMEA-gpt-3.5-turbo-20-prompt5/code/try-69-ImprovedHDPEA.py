import numpy as np

class ImprovedHDPEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.init_pop_size = 10
        self.min_pop_size = 5
        self.max_pop_size = 20
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

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        t = 0

        while t < self.budget:
            new_population = np.zeros((self.pop_size, self.dim))
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
                    new_population[i] = sa_mutation(population[i], best_solution, t)
                t += 1

            # Adaptive control for population size
            success_rate = np.sum(fitness < np.array([func(individual) for individual in new_population])) / self.pop_size
            if success_rate > 0.6 and self.pop_size < self.max_pop_size:
                self.pop_size += 1
            elif success_rate < 0.4 and self.pop_size > self.min_pop_size:
                self.pop_size -= 1

            population = new_population

        return best_solution