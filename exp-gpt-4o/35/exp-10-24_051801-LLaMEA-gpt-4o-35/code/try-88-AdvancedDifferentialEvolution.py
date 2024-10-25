import numpy as np

class AdvancedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50  # Adjusted for balance between exploration and exploitation
        self.inertia = 0.5  # Dynamic inertia for adaptive search
        self.c1 = 1.5  # Lower cognitive weight for broadened exploration
        self.c2 = 1.5  # Balanced social weight to prevent local optima
        self.mutation_factor = 0.6  # Increased mutation factor for diversity
        self.crossover_rate = 0.9  # Higher crossover rate for robust exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_strategy_threshold = 0.3  # Threshold for fitness-based mutation strategy

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.full(self.pop_size, float('inf'))
        best_idx = None
        eval_count = 0

        while eval_count < self.budget:
            for i in range(self.pop_size):
                fitness[i] = func(population[i])
                eval_count += 1

                if eval_count >= self.budget:
                    break

            if eval_count >= self.budget:
                break

            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant_vector = population[a] + self.mutation_factor * (population[b] - population[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, population[i])
                trial_fitness = func(trial_vector)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial_vector.copy()

                if eval_count >= self.budget:
                    break

            # Adaptive mutation strategy
            if np.random.rand() < self.mutation_strategy_threshold:
                for i in range(self.pop_size):
                    if fitness[i] > np.median(fitness):
                        r1, r2 = np.random.choice([j for j in range(self.pop_size) if j != i], 2, replace=False)
                        adaptive_mutant = population[i] + self.mutation_factor * (population[r1] - population[r2])
                        adaptive_mutant = np.clip(adaptive_mutant, self.lower_bound, self.upper_bound)
                        adaptive_fitness = func(adaptive_mutant)
                        eval_count += 1

                        if adaptive_fitness < fitness[i]:
                            fitness[i] = adaptive_fitness
                            population[i] = adaptive_mutant.copy()

                        if eval_count >= self.budget:
                            break

        return population[best_idx]