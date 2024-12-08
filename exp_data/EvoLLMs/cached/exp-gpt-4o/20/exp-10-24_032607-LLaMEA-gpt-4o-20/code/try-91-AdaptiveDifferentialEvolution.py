import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 8 * dim  # Increased population size for better diversity
        self.F = 0.5  # Lower differential weight for stability
        self.CR = 0.85  # High crossover probability for exploration
        self.population = None
        self.best_solution = None
        self.best_fitness = np.inf
        self.mutation_strategy = 'rand'
        self.adaptive_factor = 0.93  # Slightly adjusted adaptive factor
        self.diversity_threshold = 0.2  # Increased diversity threshold
        self.local_search_prob = 0.1  # Probability of applying local search

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

    def local_search(self, individual, func):
        perturbation = np.random.normal(0, 0.1, self.dim)
        candidate = np.clip(individual + perturbation, self.bounds[0], self.bounds[1])
        candidate_fitness = func(candidate)
        return candidate, candidate_fitness

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

                if np.random.rand() < self.local_search_prob:
                    candidate, candidate_fitness = self.local_search(trial, func)
                    func_calls += 1
                    if candidate_fitness < trial_fitness:
                        self.population[i] = candidate
                        if candidate_fitness < self.best_fitness:
                            self.best_fitness = candidate_fitness
                            self.best_solution = candidate.copy()

            historical_fitness.append(self.best_fitness)
            if len(historical_fitness) > 10:
                historical_fitness.pop(0)
                fitness_improvement = np.std(historical_fitness)
                if fitness_improvement < 1e-5:
                    self.F *= self.adaptive_factor
                    self.CR *= self.adaptive_factor
                    self.F = np.clip(self.F, 0.3, 0.8)
                    self.CR = np.clip(self.CR, 0.5, 0.95)

            if stagnation_counter > self.population_size:
                self.mutation_strategy = 'best'
            else:
                self.mutation_strategy = 'rand'

            if func_calls > self.budget * 0.75 and stagnation_counter > self.population_size * 0.25:
                for i in range(self.population_size):
                    if func_calls >= self.budget:
                        break
                    random_walk_solution = self.population[i] + np.random.normal(0, 0.2, self.dim)
                    random_walk_solution = np.clip(random_walk_solution, self.bounds[0], self.bounds[1])
                    random_walk_fitness = func(random_walk_solution)
                    func_calls += 1
                    if random_walk_fitness < self.best_fitness:
                        self.best_fitness = random_walk_fitness
                        self.best_solution = random_walk_solution.copy()

    def __call__(self, func):
        self.optimize(func)
        return self.best_solution