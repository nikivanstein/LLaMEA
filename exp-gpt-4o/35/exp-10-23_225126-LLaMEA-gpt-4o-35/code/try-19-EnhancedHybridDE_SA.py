import numpy as np

class EnhancedHybridDE_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 30
        self.population = np.random.uniform(-5.0, 5.0, (self.initial_pop_size, dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.visited_points = 0
        self.adaptive_rate = 0.95

    def __call__(self, func):
        F_base = 0.6  # Base Mutation factor with a slight increase
        CR = 0.85  # Crossover rate adjusted for balance
        temp = 1.0  # Initial temperature for Simulated Annealing
        
        while self.visited_points < self.budget:
            current_pop_size = max(5, int(self.initial_pop_size * (1 - self.visited_points / self.budget)))
            new_population = np.copy(self.population[:current_pop_size])
            for i in range(current_pop_size):
                if self.visited_points >= self.budget:
                    break

                # Dynamic Mutation factor based on iteration
                F = F_base + (0.8 - F_base) * (1 - (self.visited_points / self.budget))

                # Differential Evolution Step
                indices = np.random.choice(current_pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(current_pop_size, 3, replace=False)
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
                if new_fitness < self.fitness[i] or np.random.rand() < np.exp((self.fitness[i] - new_fitness) / (temp * self.adaptive_rate)):
                    new_population[i] = crossover_vector
                    self.fitness[i] = new_fitness

            # Adaptive temperature reduction
            temp *= 0.90
            self.population[:current_pop_size] = new_population

        # Return the best found solution
        best_idx = np.argmin(self.fitness[:current_pop_size])
        return self.population[best_idx], self.fitness[best_idx]