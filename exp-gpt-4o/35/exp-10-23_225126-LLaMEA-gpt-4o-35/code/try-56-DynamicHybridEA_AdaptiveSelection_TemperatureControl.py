import numpy as np

class DynamicHybridEA_AdaptiveSelection_TemperatureControl:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 40  # Increased initial population size for diversity
        self.population = np.random.uniform(-5.0, 5.0, (self.initial_pop_size, dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.visited_points = 0

    def __call__(self, func):
        F_min = 0.3  # Adjusted mutation factor range
        F_max = 0.8
        CR_min = 0.5  # Adjusted crossover range
        CR_max = 0.9
        temp = 1.2  # Start with a slightly higher temperature

        while self.visited_points < self.budget:
            pop_size = max(10, int(self.initial_pop_size * (1 - (self.visited_points / (1.5 * self.budget)))))  # Slower population reduction
            new_population = np.copy(self.population[:pop_size])

            for i in range(pop_size):
                if self.visited_points >= self.budget:
                    break

                CR = CR_min + (CR_max - CR_min) * (1 - np.exp(-self.visited_points / self.budget))  # Exponential decay
                F = F_min + np.random.beta(2, 5) * (F_max - F_min)  # Beta distribution for stochasticity

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

            elite_idx = np.argmin(self.fitness)
            new_population[0] = self.population[elite_idx]

            # Adaptive cooling with dynamic control
            temp *= 0.85 + 0.15 * (1 - np.cos(np.pi * self.visited_points / self.budget))
            self.population[:pop_size] = new_population

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]