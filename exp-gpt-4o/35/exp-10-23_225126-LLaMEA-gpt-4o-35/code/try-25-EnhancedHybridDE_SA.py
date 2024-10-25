import numpy as np

class EnhancedHybridDE_SA:
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

        while self.visited_points < self.budget:
            # Dynamically adjust population size
            pop_size = max(self.initial_pop_size // 2, 5)
            new_population = np.copy(self.population[:pop_size])
            for i in range(pop_size):
                if self.visited_points >= self.budget:
                    break

                # Dynamic Mutation factor based on iteration
                F = F_base + (0.9 - F_base) * (1 - (self.visited_points / self.budget))

                # Differential Evolution Step
                indices = np.random.choice(pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(pop_size, 3, replace=False)
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

                # Simulated Annealing acceptance criterion
                if new_fitness < self.fitness[i] or np.random.rand() < np.exp((self.fitness[i] - new_fitness) / temp):
                    new_population[i] = crossover_vector
                    self.fitness[i] = new_fitness

            # Adaptive temperature reduction with periodic resets
            temp *= 0.93
            if self.visited_points % 100 == 0:
                temp = max(0.5, temp)  # Avoid too low temperatures

            # Intelligent restart based on diversity
            if np.std(self.fitness[:pop_size]) < 0.01:
                self.population = np.random.uniform(-5.0, 5.0, (self.initial_pop_size, self.dim))
                self.fitness = np.full(self.initial_pop_size, np.inf)
                continue

            self.population = new_population

        # Return the best found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]