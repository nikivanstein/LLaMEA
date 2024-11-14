import numpy as np

class ImprovedHDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.cr = 0.9
        self.f = 0.8
        self.alpha = 0.9
        self.sigma = 0.1
        self.adaptive_param = 0.1  # Adaptive control parameter
        self.learning_rate = 0.05  # Dynamic learning rate
        self.inertia_weight = 0.5
        self.c1 = 1.5
        self.c2 = 1.5

    def __call__(self, func):
        def pso_mutate(target, best):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            velocity = self.inertia_weight * velocity + self.c1 * r1 * (best - target) + self.c2 * r2 * (best - target)
            return target + velocity

        def sa_mutation(candidate, best, t):
            self.sigma = max(0.01, min(0.5, self.sigma + np.random.normal(0, self.adaptive_param)))  # Adaptive control
            self.learning_rate = max(0.01, min(0.1, self.learning_rate + np.random.normal(0, self.adaptive_param)))  # Dynamic learning rate adjustment
            return candidate + (self.sigma + self.learning_rate) * np.exp(-t * self.alpha) * np.random.normal(0, 1, self.dim)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        velocity = np.zeros((self.pop_size, self.dim))
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
                    new_population[i] = pso_mutate(population[i], best_solution)
                t += 1

            population = new_population

        return best_solution