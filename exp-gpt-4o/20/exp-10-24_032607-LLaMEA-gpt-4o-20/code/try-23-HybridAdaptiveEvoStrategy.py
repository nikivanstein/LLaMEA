import numpy as np

class HybridAdaptiveEvoStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 12 * dim
        self.F = 0.6  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = np.inf
        self.elitism_rate = 0.1  # Proportion of elites retained
        self.mutation_strategy = 'rand'
        self.adaptive_factor = 0.9  # Adaptive factor for dynamic adjustments

    def initialize_population(self):
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
    
    def evaluate_population(self, func):
        fitness = np.array([func(ind) for ind in self.population])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_solution = self.population[min_idx].copy()
        return fitness

    def mutate(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        np.random.shuffle(indices)
        a, b, c = indices[:3]
        if self.mutation_strategy == 'best':
            target_vector = self.best_solution
        else:
            target_vector = self.population[a]
        return target_vector + self.F * (self.population[b] - self.population[c])

    def crossover(self, target, mutant):
        j_rand = np.random.randint(0, self.dim)
        trial = np.array([mutant[j] if np.random.rand() < self.CR or j == j_rand else target[j] for j in range(self.dim)])
        return np.clip(trial, self.bounds[0], self.bounds[1])

    def optimize(self, func):
        self.initialize_population()
        func_calls = 0
        historical_fitness = []

        while func_calls < self.budget:
            fitness = self.evaluate_population(func)
            func_calls += self.population_size

            new_population = []
            for i in range(self.population_size):
                if func_calls >= self.budget:
                    break
                target = self.population[i]
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)
                trial_fitness = func(trial)
                func_calls += 1
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                else:
                    new_population.append(target)
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial.copy()

            self.population = np.array(new_population)
            historical_fitness.append(self.best_fitness)
            if len(historical_fitness) > 20:
                historical_fitness.pop(0)
                fitness_improvement = np.std(historical_fitness)
                if fitness_improvement < 1e-6:
                    self.F *= self.adaptive_factor
                    self.CR *= self.adaptive_factor
                    self.F = np.clip(self.F, 0.3, 0.9)
                    self.CR = np.clip(self.CR, 0.5, 0.95)

            if func_calls > self.budget * 0.8:
                elite_count = int(self.elitism_rate * self.population_size)
                elite_indices = np.argsort(fitness)[:elite_count]
                elite_solutions = self.population[elite_indices]
                for i in range(self.population_size):
                    if func_calls >= self.budget:
                        break
                    random_walk_solution = self.population[i] + np.random.normal(0, 0.05, self.dim)
                    random_walk_solution = np.clip(random_walk_solution, self.bounds[0], self.bounds[1])
                    random_walk_fitness = func(random_walk_solution)
                    func_calls += 1
                    if random_walk_fitness < self.best_fitness:
                        self.best_fitness = random_walk_fitness
                        self.best_solution = random_walk_solution.copy()
                self.population = np.concatenate((elite_solutions, self.population[:self.population_size - elite_count]))

    def __call__(self, func):
        self.optimize(func)
        return self.best_solution