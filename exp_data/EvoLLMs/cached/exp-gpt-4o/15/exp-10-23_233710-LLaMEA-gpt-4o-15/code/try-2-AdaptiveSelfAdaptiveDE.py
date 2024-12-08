import numpy as np

class AdaptiveSelfAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(5, dim * 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scaling_factors = np.random.uniform(0.5, 1.0, self.population_size)
        self.crossover_rates = np.random.uniform(0.7, 1.0, self.population_size)
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
                trial = self.mutation_strategies[strategy_idx](population, i, best, self.scaling_factors[i])
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                    self.strategy_weights[strategy_idx] += 1
                    # Self-adaptive parameter adjustment
                    self.scaling_factors[i] = np.minimum(1.0, self.scaling_factors[i] + 0.1)
                    self.crossover_rates[i] = np.minimum(1.0, self.crossover_rates[i] + 0.05)
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    # Self-adaptive parameter adjustment
                    self.scaling_factors[i] = np.maximum(0.5, self.scaling_factors[i] - 0.1)
                    self.crossover_rates[i] = np.maximum(0.7, self.crossover_rates[i] - 0.05)
            
            population = new_population
            fitness = new_fitness
            best_idx = np.argmin(fitness)
            best = population[best_idx]
        
        return best

    def de_rand_1_bin(self, population, idx, best, scale_factor):
        a, b, c = population[np.random.choice(range(self.population_size), 3, replace=False)]
        mutant = a + scale_factor * (b - c)
        return self.binomial_crossover(population[idx], mutant, self.crossover_rates[idx])

    def de_best_1_bin(self, population, idx, best, scale_factor):
        a, b = population[np.random.choice(range(self.population_size), 2, replace=False)]
        mutant = best + scale_factor * (a - b)
        return self.binomial_crossover(population[idx], mutant, self.crossover_rates[idx])

    def de_rand_to_best_1_bin(self, population, idx, best, scale_factor):
        a, b = population[np.random.choice(range(self.population_size), 2, replace=False)]
        mutant = population[idx] + scale_factor * (best - population[idx]) + scale_factor * (a - b)
        return self.binomial_crossover(population[idx], mutant, self.crossover_rates[idx])

    def binomial_crossover(self, target, mutant, crossover_rate):
        trial = np.empty_like(target)
        jrand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() < crossover_rate or j == jrand:
                trial[j] = mutant[j]
            else:
                trial[j] = target[j]
        return trial