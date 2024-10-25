import numpy as np

class RefinedAdaptiveSwarmEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 15 * dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.strategy_probs = [0.4, 0.4, 0.2]
        self.mutation_factor = 0.9
        self.crossover_rate = 0.85
        self.memory = np.zeros((10, self.dim))
        self.stagnation_limit = int(self.pop_size * 0.1)

    def select_strategy(self):
        return np.random.choice(['rand_1_bin', 'current_to_best_1_bin', 'memory_mutation'], p=self.strategy_probs)

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.fitness[i] == np.inf:
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
        elif strategy == 'memory_mutation':
            x = self.population[idx]
            memory_idx = np.random.randint(0, len(self.memory))
            mutant = x + self.mutation_factor * (self.memory[memory_idx] - x)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        crossover_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_points):
            crossover_points[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_points, mutant, target)
        return trial

    def update_strategy_probs(self, success, strategy):
        if success:
            if strategy == 'rand_1_bin':
                self.strategy_probs = [0.45, 0.35, 0.2]
            elif strategy == 'current_to_best_1_bin':
                self.strategy_probs = [0.35, 0.45, 0.2]
            else:
                self.strategy_probs = [0.4, 0.4, 0.2]
        else:
            if strategy == 'rand_1_bin':
                self.strategy_probs = [0.35, 0.45, 0.2]
            elif strategy == 'current_to_best_1_bin':
                self.strategy_probs = [0.45, 0.35, 0.2]
            else:
                self.strategy_probs = [0.4, 0.4, 0.2]

    def adaptive_restart(self):
        restart_threshold = self.stagnation_limit
        if self.stagnation_counter > restart_threshold:
            self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
            self.fitness = np.full(self.pop_size, np.inf)
            self.stagnation_counter = 0

    def dynamic_population_control(self):
        if self.best_fitness < np.inf:
            self.pop_size = max(10, int(self.pop_size * 0.9))
            self.population = self.population[:self.pop_size]
            self.fitness = self.fitness[:self.pop_size]

    def optimize(self, func):
        evaluations = 0
        self.stagnation_counter = 0
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
                        self.stagnation_counter = 0
                    self.update_strategy_probs(True, strategy)
                else:
                    new_population[i] = self.population[i]
                    self.update_strategy_probs(False, strategy)

                if evaluations >= self.budget:
                    break

            if self.best_fitness >= previous_best_fitness:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                previous_best_fitness = self.best_fitness
            
            self.adaptive_restart()
            self.dynamic_population_control()
            self.population = new_population

    def __call__(self, func):
        self.optimize(func)
        return self.best_solution, self.best_fitness