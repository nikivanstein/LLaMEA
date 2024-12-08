import numpy as np

class ImprovedHybridDE_SA_Adaptive:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 25  # Adjusted population size for better exploration
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.visited_points = 0

    def __call__(self, func):
        F_base = 0.6  # Adjusted base Mutation factor for diversity
        CR = 0.85  # Adjusted Crossover rate for better exploitation
        temp = 0.9  # Initial temperature for Simulated Annealing

        while self.visited_points < self.budget:
            new_population = np.copy(self.population)
            for i in range(self.pop_size):
                if self.visited_points >= self.budget:
                    break

                # Dynamic Mutation factor with additional randomness
                F = F_base + (0.9 - F_base) * np.random.rand() * (1 - (self.visited_points / self.budget))

                # Differential Evolution Step
                indices = np.random.choice(self.pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)

                # Crossover with adaptive randomness
                crossover_vector = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        crossover_vector[j] = mutant[j]

                # Selection
                new_fitness = func(crossover_vector)
                self.visited_points += 1

                # Simulated Annealing acceptance criterion with adaptive cooling schedule
                if new_fitness < self.fitness[i] or np.random.rand() < np.exp((self.fitness[i] - new_fitness) / (temp + 0.1)):
                    new_population[i] = crossover_vector
                    self.fitness[i] = new_fitness

            # Adaptive temperature reduction with improved cooling schedule
            temp *= 0.92
            self.population = new_population

        # Return the best found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]