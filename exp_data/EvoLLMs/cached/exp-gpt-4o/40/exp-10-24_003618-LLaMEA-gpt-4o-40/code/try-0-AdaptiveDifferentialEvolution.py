import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 10 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.strategy_probs = [0.5, 0.5]  # Start with equal probability for strategies
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def select_strategy(self):
        return np.random.choice(['rand_1_bin', 'current_to_best_1_bin'], p=self.strategy_probs)

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.fitness[i] == np.inf:  # Evaluate only if not yet evaluated
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()

    def mutation(self, idx, strategy):
        if strategy == 'rand_1_bin':
            indices = np.random.choice(self.pop_size, 3, replace=False)
            x0, x1, x2 = self.population[indices]
            mutant = x0 + self.mutation_factor * (x1 - x2)
        elif strategy == 'current_to_best_1_bin':
            x = self.population[idx]
            indices = np.random.choice(self.pop_size, 2, replace=False)
            x1, x2 = self.population[indices]
            mutant = x + self.mutation_factor * (self.best_solution - x) + self.mutation_factor * (x1 - x2)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        crossover_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_points):
            crossover_points[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_points, mutant, target)
        return trial

    def optimize(self, func):
        evaluations = 0
        while evaluations < self.budget:
            self.evaluate_population(func)
            new_population = np.empty_like(self.population)

            for i in range(self.pop_size):
                strategy = self.select_strategy()
                mutant = self.mutation(i, strategy)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    new_population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial
                    # Update strategy probabilities
                    self.strategy_probs = [0.6, 0.4] if strategy == 'rand_1_bin' else [0.4, 0.6]
                else:
                    new_population[i] = self.population[i]

                if evaluations >= self.budget:
                    break

            self.population = new_population

    def __call__(self, func):
        self.optimize(func)
        return self.best_solution, self.best_fitness