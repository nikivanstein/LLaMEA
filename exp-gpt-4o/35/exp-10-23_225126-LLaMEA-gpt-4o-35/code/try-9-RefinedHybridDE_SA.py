import numpy as np

class RefinedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 25
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.visited_points = 0

    def __call__(self, func):
        F_base = 0.6  # Base Mutation factor
        CR = 0.8  # Crossover rate
        temp = 1.5  # Initial temperature for Simulated Annealing
        elite_rate = 0.1  # Proportion of elite candidates

        while self.visited_points < self.budget:
            new_population = np.copy(self.population)
            elite_count = int(self.pop_size * elite_rate)
            elite_indices = np.argsort(self.fitness)[:elite_count]

            for i in range(self.pop_size):
                if self.visited_points >= self.budget:
                    break

                # Dynamic Mutation factor with enhanced scaling
                F = F_base + (0.8 - F_base) * (1 - (self.visited_points / self.budget)**1.5)

                # Select elite individuals for variation
                if i in elite_indices:
                    indices = np.random.choice(elite_indices, 2, replace=False)
                else:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)

                # Crossover with randomness for exploration
                crossover_vector = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        crossover_vector[j] = mutant[j]

                # Selection with elitism consideration
                new_fitness = func(crossover_vector)
                self.visited_points += 1

                # Simulated Annealing acceptance criterion
                if new_fitness < self.fitness[i] or np.random.rand() < np.exp((self.fitness[i] - new_fitness) / temp):
                    new_population[i] = crossover_vector
                    self.fitness[i] = new_fitness

            # Adaptive temperature reduction with a slower decay
            temp *= 0.95
            self.population = new_population

        # Return the best found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]