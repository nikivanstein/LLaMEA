import numpy as np

class AdaptiveHybridDE_ES_MutationCooling:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 30
        self.population = np.random.uniform(-5.0, 5.0, (self.initial_pop_size, dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.visited_points = 0

    def __call__(self, func):
        F_min = 0.3
        F_max = 0.8
        CR_min = 0.5
        CR_max = 0.9
        temp = 1.0

        while self.visited_points < self.budget:
            pop_size = max(5, int(self.initial_pop_size * (1 - (self.visited_points / self.budget))))
            new_population = np.copy(self.population[:pop_size])

            for i in range(pop_size):
                if self.visited_points >= self.budget:
                    break

                CR = CR_min + (CR_max - CR_min) * np.exp(-self.visited_points / self.budget)
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

            # Mutation cooling
            if self.visited_points < self.budget * 0.5:
                F_max += (0.85 - F_max) * (self.visited_points / self.budget)
            temp *= 0.8 + 0.15 * np.random.rand()
            self.population[:pop_size] = new_population

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]