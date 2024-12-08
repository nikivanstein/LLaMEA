import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factors = np.linspace(0.4, 1.0, num=4)  # dynamic range
        self.crossover_rate = 0.9
        self.mutation_strategies = [
            self.de_rand_1_bin,
            self.de_best_1_bin,
            self.de_rand_to_best_1_bin
        ]
        self.strategy_weights = np.ones(len(self.mutation_strategies))

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
                trial = self.mutation_strategies[strategy_idx](population, i, best)
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

    def de_rand_1_bin(self, population, idx, best):
        a, b, c = population[np.random.choice(range(self.population_size), 3, replace=False)]
        mutant = a + np.random.uniform(0.4, 1.0) * (b - c)  # dynamic scaling
        return self.binomial_crossover(population[idx], mutant)

    def de_best_1_bin(self, population, idx, best):
        a, b = population[np.random.choice(range(self.population_size), 2, replace=False)]
        mutant = best + np.random.uniform(0.4, 1.0) * (a - b)  # dynamic scaling
        return self.binomial_crossover(population[idx], mutant)

    def de_rand_to_best_1_bin(self, population, idx, best):
        a, b = population[np.random.choice(range(self.population_size), 2, replace=False)]
        mutant = population[idx] + np.random.uniform(0.4, 1.0) * (best - population[idx]) + \
                 np.random.uniform(0.4, 1.0) * (a - b)  # dynamic scaling
        return self.binomial_crossover(population[idx], mutant)

    def binomial_crossover(self, target, mutant):
        trial = np.empty_like(target)
        jrand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.crossover_rate or j == jrand:
                trial[j] = mutant[j]
            else:
                trial[j] = target[j]
        return trial