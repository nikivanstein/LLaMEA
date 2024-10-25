import numpy as np

class EnhancedHybridDE_SA_RS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 25  # Increased population size
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.visited_points = 0
        self.best_fitness = np.inf
        self.best_solution = None

    def __call__(self, func):
        F_base = 0.6  # Adjusted Base Mutation factor
        CR = 0.85  # Adjusted Crossover rate
        temp = 1.5  # Adjusted Initial temperature for Simulated Annealing

        while self.visited_points < self.budget:
            new_population = np.copy(self.population)
            for i in range(self.pop_size):
                if self.visited_points >= self.budget:
                    break

                # Adaptive Mutation factor
                F = F_base + (0.8 - F_base) * (1 - (self.visited_points / self.budget))

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

                # Random Search
                if np.random.rand() < 0.1:
                    crossover_vector = np.random.uniform(-5.0, 5.0, self.dim)

                # Selection
                new_fitness = func(crossover_vector)
                self.visited_points += 1

                # Simulated Annealing acceptance criterion
                if new_fitness < self.fitness[i] or np.random.rand() < np.exp((self.fitness[i] - new_fitness) / temp):
                    new_population[i] = crossover_vector
                    self.fitness[i] = new_fitness

                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_solution = crossover_vector

            # Adaptive temperature reduction
            temp *= 0.95
            self.population = new_population

        return self.best_solution, self.best_fitness