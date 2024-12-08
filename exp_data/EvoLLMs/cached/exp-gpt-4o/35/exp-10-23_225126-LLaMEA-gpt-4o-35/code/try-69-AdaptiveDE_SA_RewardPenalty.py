import numpy as np

class AdaptiveDE_SA_RewardPenalty:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = 30
        self.population = np.random.uniform(-5.0, 5.0, (self.initial_pop_size, dim))
        self.fitness = np.full(self.initial_pop_size, np.inf)
        self.visited_points = 0
        self.reward_penalty = np.zeros(self.initial_pop_size)

    def __call__(self, func):
        F_min = 0.4
        F_max = 0.9
        CR_min = 0.5
        CR_max = 0.9
        temp = 1.0
        adjustment_factor = 0.05

        while self.visited_points < self.budget:
            pop_size = max(5, int(self.initial_pop_size * (1 - (self.visited_points / self.budget))))
            new_population = np.copy(self.population[:pop_size])

            for i in range(pop_size):
                if self.visited_points >= self.budget:
                    break

                CR = CR_min + (CR_max - CR_min) * (1 - (self.visited_points / self.budget))
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
                    self.reward_penalty[i] = max(0, self.reward_penalty[i] - adjustment_factor)
                else:
                    self.reward_penalty[i] += adjustment_factor

            best_idx = np.argmin(self.fitness)
            new_population[0] = self.population[best_idx]

            # Adaptive cooling
            temp *= 0.85 + 0.05 * (self.visited_points / self.budget)
            self.population[:pop_size] = new_population

            # Reward-Penalty adjustment
            for i in range(pop_size):
                if np.random.rand() < self.reward_penalty[i]:
                    self.population[i] = np.random.uniform(-5.0, 5.0, self.dim)
                    self.fitness[i] = func(self.population[i])

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]