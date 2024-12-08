import numpy as np

class EnhancedHybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 12 * dim  # Increased population size for diversity
        self.F = 0.9  # Slightly increased differential weight for explorative search
        self.CR = 0.85  # Lower crossover probability for diversity
        self.population = None
        self.best_solution = None
        self.best_fitness = np.inf
        self.mutation_strategy = 'rand'  # Default mutation strategy

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
            return self.best_solution + self.F * (self.population[b] - self.population[c])
        else:
            return self.population[a] + self.F * (self.population[b] - self.population[c])

    def crossover(self, target, mutant):
        j_rand = np.random.randint(0, self.dim)
        trial = np.array([mutant[j] if np.random.rand() < self.CR or j == j_rand else target[j] for j in range(self.dim)])
        return np.clip(trial, self.bounds[0], self.bounds[1])

    def optimize(self, func):
        self.initialize_population()
        func_calls = 0
        stagnation_counter = 0

        while func_calls < self.budget:
            for i in range(self.population_size):
                if func_calls >= self.budget:
                    break
                target = self.population[i]
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)
                trial_fitness = func(trial)
                func_calls += 1
                if trial_fitness < func(target):
                    self.population[i] = trial
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            # Adaptive mutation strategy adjustment
            if stagnation_counter > self.population_size:
                self.mutation_strategy = 'best'
            else:
                self.mutation_strategy = 'rand'

            # Stochastic tunneling to escape local optima
            if func_calls > self.budget * 0.8 and stagnation_counter > self.population_size * 0.6:
                for i in range(self.population_size):
                    if func_calls >= self.budget:
                        break
                    tunneling_factor = np.random.normal(0, 0.1, self.dim)
                    tunneled_solution = self.population[i] + tunneling_factor
                    tunneled_solution = np.clip(tunneled_solution, self.bounds[0], self.bounds[1])
                    tunneled_fitness = func(tunneled_solution)
                    func_calls += 1
                    if tunneled_fitness < self.best_fitness:
                        self.best_fitness = tunneled_fitness
                        self.best_solution = tunneled_solution.copy()

    def __call__(self, func):
        self.optimize(func)
        return self.best_solution