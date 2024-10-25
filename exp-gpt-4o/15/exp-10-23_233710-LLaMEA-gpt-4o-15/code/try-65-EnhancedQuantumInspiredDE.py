import numpy as np

class EnhancedQuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factors = [0.4, 0.6, 0.8, 1.0]
        self.crossover_rate = 0.9
        self.mutation_strategies = [
            self.de_rand_1_bin,
            self.de_best_1_bin,
            self.de_rand_to_best_1_bin
        ]
        self.strategy_weights = np.ones(len(self.mutation_strategies))
        self.chaotic_sequence = self.init_chaotic_sequence()

    def init_chaotic_sequence(self):
        sequence = np.empty(self.budget)
        sequence[0] = np.random.rand()
        for i in range(1, self.budget):
            sequence[i] = 4 * sequence[i-1] * (1 - sequence[i-1])
        return sequence

    def __call__(self, func):
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        best_idx = np.argmin(fitness)
        best = population[best_idx]

        while eval_count < self.budget:
            new_population = np.empty_like(population)
            new_fitness = np.empty(self.population_size)

            for i in range(self.population_size):
                strategy_idx = np.random.choice(
                    len(self.mutation_strategies), p=self.strategy_weights / self.strategy_weights.sum()
                )
                trial = self.mutation_strategies[strategy_idx](population, i, best, eval_count)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                    self.strategy_weights[strategy_idx] += 0.5
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    self.strategy_weights[strategy_idx] *= 0.7

            population = new_population
            fitness = new_fitness
            best_idx = np.argmin(fitness)
            best = population[best_idx]

        return best

    def de_rand_1_bin(self, population, idx, best, eval_count):
        a, b, c = population[np.random.choice(range(self.population_size), 3, replace=False)]
        chaotic_factor = self.chaotic_sequence[eval_count % self.budget]
        mutant = a + chaotic_factor * np.random.choice(self.scaling_factors) * (b - c)
        return self.binomial_crossover(population[idx], mutant, eval_count)

    def de_best_1_bin(self, population, idx, best, eval_count):
        a, b = population[np.random.choice(range(self.population_size), 2, replace=False)]
        chaotic_factor = self.chaotic_sequence[eval_count % self.budget]
        mutant = best + chaotic_factor * np.random.choice(self.scaling_factors) * (a - b)
        return self.binomial_crossover(population[idx], mutant, eval_count)

    def de_rand_to_best_1_bin(self, population, idx, best, eval_count):
        a, b = population[np.random.choice(range(self.population_size), 2, replace=False)]
        chaotic_factor = self.chaotic_sequence[eval_count % self.budget]
        mutant = population[idx] + chaotic_factor * np.random.choice(self.scaling_factors) * (best - population[idx]) + \
                 chaotic_factor * np.random.choice(self.scaling_factors) * (a - b)
        return self.binomial_crossover(population[idx], mutant, eval_count)

    def binomial_crossover(self, target, mutant, eval_count):
        trial = np.empty_like(target)
        jrand = np.random.randint(self.dim)
        chaotic_crossover_rate = self.crossover_rate * self.chaotic_sequence[eval_count % self.budget]
        for j in range(self.dim):
            if np.random.rand() < chaotic_crossover_rate or j == jrand:
                trial[j] = mutant[j]
            else:
                trial[j] = target[j]
        return trial