import numpy as np

class ImprovedHybridDE_SA_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 24  # Slightly larger population
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.visited_points = 0

    def __call__(self, func):
        F_base = 0.6  # Increased base Mutation factor
        CR = 0.85  # Slightly reduced Crossover rate
        temp = 1.2  # Higher initial temperature

        while self.visited_points < self.budget:
            new_population = np.copy(self.population)
            for i in range(self.pop_size):
                if self.visited_points >= self.budget:
                    break

                # Adaptive Mutation factor based on fitness rank
                rank = np.argsort(self.fitness)
                F = F_base + (0.7 - F_base) * (rank[i] / self.pop_size)

                # Dynamic Crossover Step
                indices = np.random.choice(self.pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)

                crossover_vector = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        crossover_vector[j] = mutant[j]

                # Selection and Acceptance
                new_fitness = func(crossover_vector)
                self.visited_points += 1

                # Enhanced Simulated Annealing acceptance criterion
                if new_fitness < self.fitness[i] or np.random.rand() < np.exp((self.fitness[i] - new_fitness) / (temp + 1e-10)):
                    new_population[i] = crossover_vector
                    self.fitness[i] = new_fitness

            # Enhanced Adaptive temperature reduction
            temp *= 0.95
            self.population = new_population

        # Return the best found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]