import numpy as np

class HybridAdaptiveEvoStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 15 * dim
        self.F = 0.6  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.best_solution = None
        self.best_fitness = np.inf
        self.mutation_strategy = 'rand'
        self.adaptive_factor = 0.9  # Adaptive factor for dynamic adjustments
        self.temperature = 1.0  # Initial temperature for simulated annealing

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

    def simulated_annealing(self, solution, func):
        perturbation = np.random.normal(0, self.temperature, self.dim)
        new_solution = np.clip(solution + perturbation, self.bounds[0], self.bounds[1])
        new_fitness = func(new_solution)
        if new_fitness < self.best_fitness or np.random.rand() < np.exp((self.best_fitness - new_fitness) / self.temperature):
            self.best_solution = new_solution
            self.best_fitness = new_fitness

    def optimize(self, func):
        self.initialize_population()
        func_calls = 0
        stagnation_counter = 0
        historical_fitness = []

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

            historical_fitness.append(self.best_fitness)
            if len(historical_fitness) > 20:
                historical_fitness.pop(0)
                fitness_improvement = np.std(historical_fitness)
                if fitness_improvement < 1e-6:
                    self.F *= self.adaptive_factor
                    self.CR *= self.adaptive_factor
                    self.F = np.clip(self.F, 0.2, 0.9)
                    self.CR = np.clip(self.CR, 0.4, 0.95)

            if stagnation_counter > self.population_size:
                self.mutation_strategy = 'best'
            else:
                self.mutation_strategy = 'rand'

            if func_calls > self.budget * 0.8 and stagnation_counter > self.population_size * 0.5:
                for i in range(self.population_size):
                    if func_calls >= self.budget:
                        break
                    self.simulated_annealing(self.population[i], func)
                    func_calls += 1
                self.temperature *= 0.95  # Cooling schedule

    def __call__(self, func):
        self.optimize(func)
        return self.best_solution