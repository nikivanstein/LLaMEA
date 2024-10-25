import numpy as np

class RefinedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 20
        self.population = np.random.uniform(-5.0, 5.0, (self.initial_pop_size, dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.visited_points = 0

    def __call__(self, func):
        F_base = 0.5  # Base Mutation factor
        CR_base = 0.9  # Base Crossover rate
        temp = 1.0  # Initial temperature for Simulated Annealing
        adaptive_rate = 0.95  # Rate of population resizing

        while self.visited_points < self.budget:
            current_pop_size = len(self.population)
            new_population = np.copy(self.population)
            for i in range(current_pop_size):
                if self.visited_points >= self.budget:
                    break

                # Dynamic Mutation factor based on iteration
                F = F_base + (0.8 - F_base) * (1 - (self.visited_points / self.budget))

                # Adjust Crossover Rate
                CR = CR_base * (1 - 0.3 * (self.visited_points / self.budget))

                # Differential Evolution Step
                indices = np.random.choice(current_pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(current_pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)

                # Crossover with enhanced strategy
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

            # Adaptive temperature and population reduction
            temp *= 0.92
            if np.random.rand() < adaptive_rate and current_pop_size > 5:
                new_population = new_population[:int(current_pop_size * adaptive_rate)]
                self.fitness = self.fitness[:new_population.shape[0]]

            self.population = new_population

        # Return the best found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]