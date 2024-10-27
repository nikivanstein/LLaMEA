import numpy as np

class HybridFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.alpha = 0.2  # Firefly attractiveness coefficient
        self.beta_min = 0.2  # Minimum mutation step size
        self.beta_max = 1.0  # Maximum mutation step size

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for _ in range(self.budget):
            for i in range(self.budget):
                for j in range(self.budget):
                    if fitness[j] < fitness[i]:  # Firefly attraction phase
                        attractive_force = self.alpha / (1 + np.linalg.norm(population[i] - population[j]))
                        population[i] += attractive_force * (population[j] - population[i])

                # Mutation phase
                beta = self.beta_min + (self.beta_max - self.beta_min) * np.random.rand()
                mutant = population[i] + beta * np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                mutant_fitness = func(mutant)

                if mutant_fitness < fitness[i]:  # Acceptance criterion
                    population[i] = mutant
                    fitness[i] = mutant_fitness

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx]
                best_fitness = fitness[best_idx]

        return best_solution