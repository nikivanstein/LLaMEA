import numpy as np

class EnhancedHybridDE_SA_Elitism_AdaptiveCooling_V2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 35  # Slight increase to enhance diversity
        self.population = np.random.uniform(-5.0, 5.0, (self.initial_pop_size, dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.visited_points = 0

    def __call__(self, func):
        F_min = 0.5  # Modified to provide slightly more aggressive exploration
        F_max = 0.85
        CR_min = 0.5  # Increased dynamic range for crossover rate
        CR_max = 0.98
        temp = 1.0

        while self.visited_points < self.budget:
            pop_size = max(5, int(self.initial_pop_size * (1 - (self.visited_points / self.budget))))
            new_population = np.copy(self.population[:pop_size])

            for i in range(pop_size):
                if self.visited_points >= self.budget:
                    break

                CR = CR_min + (CR_max - CR_min) * (1 - np.exp(-3 * (self.visited_points / self.budget)))  # More dynamic adjustment of CR
                F = F_min + np.random.rand() * (F_max - F_min)

                indices = np.random.choice(pop_size, 3, replace=False)
                while i in indices:
                    indices = np.random.choice(pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)

                crossover_vector = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < CR or j == j_rand:
                        crossover_vector[j] = mutant[j]

                new_fitness = func(crossover_vector)
                self.visited_points += 1

                if new_fitness < self.fitness[i] or np.random.rand() < np.exp((self.fitness[i] - new_fitness) / temp):
                    new_population[i] = crossover_vector
                    self.fitness[i] = new_fitness

            best_idx = np.argmin(self.fitness)
            new_population[0] = self.population[best_idx]

            # Improved adaptive cooling
            temp *= 0.85 + 0.05 * (np.sqrt(self.visited_points) / np.sqrt(self.budget))
            self.population[:pop_size] = new_population

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]