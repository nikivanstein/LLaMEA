import numpy as np

class RefinedSwarmOptimizer:
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
        self.strategy_probs = [0.4, 0.4, 0.2]
        self.adaptive_factor = 0.8
        self.crossover_rate = 0.9
        self.memory = np.zeros((3, self.dim))
        self.diversity_threshold = 0.15
        self.stagnation_count = 0
        self.max_stagnation = 12

    def select_strategy(self):
        return np.random.choice(['best_1_bin', 'target_to_best_1_bin', 'memory_guided'], p=self.strategy_probs)

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()

    def mutation(self, idx, strategy):
        if strategy == 'best_1_bin':
            indices = np.random.choice(np.delete(np.arange(self.pop_size), idx), 2, replace=False)
            x1, x2 = self.population[indices]
            mutant = self.best_solution + self.adaptive_factor * (x1 - x2)
        elif strategy == 'target_to_best_1_bin':
            x = self.population[idx]
            indices = np.random.choice(np.delete(np.arange(self.pop_size), idx), 2, replace=False)
            x1, x2 = self.population[indices]
            mutant = x + self.adaptive_factor * (self.best_solution - x) + self.adaptive_factor * (x1 - x2)
        elif strategy == 'memory_guided':
            x = self.population[idx]
            memory_idx = np.random.randint(0, len(self.memory))
            mutant = x + self.adaptive_factor * (self.memory[memory_idx] - x)
        else:
            mutant = self.best_solution + np.random.uniform(-1, 1, self.dim)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        crossover_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_points):
            crossover_points[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_points, mutant, target)
        return trial

    def update_strategy_probs(self, success, strategy):
        if success:
            if strategy == 'best_1_bin':
                self.strategy_probs = [0.5, 0.3, 0.2]
            elif strategy == 'target_to_best_1_bin':
                self.strategy_probs = [0.3, 0.5, 0.2]
            else:
                self.strategy_probs = [0.3, 0.3, 0.4]
        else:
            self.strategy_probs = [0.4, 0.4, 0.2]

    def diversity_measure(self):
        mean_position = np.mean(self.population, axis=0)
        diversity = np.mean(np.linalg.norm(self.population - mean_position, axis=1))
        return diversity

    def adaptive_restart(self):
        if self.diversity_measure() < self.diversity_threshold:
            self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
            self.fitness = np.full(self.pop_size, np.inf)
            self.stagnation_count = 0
        if self.best_fitness < 1e-6 or self.stagnation_count > self.max_stagnation:
            self.adaptive_factor *= 0.9
            self.stagnation_count = 0

    def optimize(self, func):
        evaluations = 0
        previous_best_fitness = self.best_fitness

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
                        self.memory = np.roll(self.memory, shift=-1, axis=0)
                        self.memory[-1] = trial
                    self.update_strategy_probs(True, strategy)
                else:
                    new_population[i] = self.population[i]
                    self.update_strategy_probs(False, strategy)

                if evaluations >= self.budget:
                    break

            if self.best_fitness >= previous_best_fitness:
                self.stagnation_count += 1
                self.adaptive_restart()
            else:
                previous_best_fitness = self.best_fitness
                self.stagnation_count = 0

            self.population = new_population

    def __call__(self, func):
        self.optimize(func)
        return self.best_solution, self.best_fitness