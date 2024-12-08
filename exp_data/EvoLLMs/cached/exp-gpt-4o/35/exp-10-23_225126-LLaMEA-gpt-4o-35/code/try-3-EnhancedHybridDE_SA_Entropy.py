import numpy as np

class EnhancedHybridDE_SA_Entropy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 25  # Increased population for more exploration
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.visited_points = 0

    def __call__(self, func):
        F_base = 0.5  # Base Mutation factor
        CR = 0.85  # Modified Crossover rate for balance
        temp = 1.0  # Initial temperature for Simulated Annealing

        while self.visited_points < self.budget:
            new_population = np.copy(self.population)
            for i in range(self.pop_size):
                if self.visited_points >= self.budget:
                    break

                # Adaptive Mutation factor using entropy
                diversity = np.var(self.population, axis=0)
                F = F_base + (0.9 - F_base) * (1 - (self.visited_points / self.budget)) * np.mean(diversity)

                # Differential Evolution Step
                indices = np.random.choice(self.pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)

                # Crossover
                crossover_vector = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        crossover_vector[j] = mutant[j]

                # Selection
                new_fitness = func(crossover_vector)
                self.visited_points += 1

                # Simulated Annealing acceptance criterion with entropy-based cooling
                if new_fitness < self.fitness[i] or np.random.rand() < np.exp((self.fitness[i] - new_fitness) / temp):
                    new_population[i] = crossover_vector
                    self.fitness[i] = new_fitness

            # Adaptive temperature reduction based on population entropy
            entropy = -np.sum(diversity * np.log(diversity + 1e-9))
            temp *= 0.93 + 0.02 * entropy
            self.population = new_population

        # Return the best found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]