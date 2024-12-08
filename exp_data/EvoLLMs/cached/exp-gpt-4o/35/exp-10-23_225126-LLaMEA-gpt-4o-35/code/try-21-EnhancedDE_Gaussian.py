import numpy as np

class EnhancedDE_Gaussian:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.population = np.random.uniform(-5.0, 5.0, (self.initial_pop_size, dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.visited_points = 0

    def __call__(self, func):
        F_base = 0.5  # Base Mutation factor
        CR = 0.9  # Crossover rate
        temp = 1.0  # Initial temperature for Simulated Annealing
        sigma = 0.1  # Standard deviation for Gaussian mutation

        while self.visited_points < self.budget:
            pop_size = self.adaptive_population_size()
            new_population = np.copy(self.population[:pop_size])
            for i in range(pop_size):
                if self.visited_points >= self.budget:
                    break

                # Dynamic Mutation factor based on iteration
                F = F_base + (0.9 - F_base) * (1 - (self.visited_points / self.budget))

                # Differential Evolution Step with Gaussian mutation
                indices = np.random.choice(pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c) + np.random.normal(0, sigma, self.dim), -5.0, 5.0)

                # Crossover
                crossover_vector = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        crossover_vector[j] = mutant[j]

                # Selection
                new_fitness = func(crossover_vector)
                self.visited_points += 1

                # Simulated Annealing acceptance criterion
                if new_fitness < self.fitness[i] or np.random.rand() < np.exp((self.fitness[i] - new_fitness) / temp):
                    new_population[i] = crossover_vector
                    self.fitness[i] = new_fitness

            # Adaptive temperature reduction
            temp *= 0.93
            self.population[:pop_size] = new_population

        # Return the best found solution
        best_idx = np.argmin(self.fitness[:pop_size])
        return self.population[best_idx], self.fitness[best_idx]

    def adaptive_population_size(self):
        # Dynamically adjust population size based on current fitness distribution
        return max(10, int(self.initial_pop_size * (1.0 - np.std(self.fitness) / np.mean(self.fitness))))