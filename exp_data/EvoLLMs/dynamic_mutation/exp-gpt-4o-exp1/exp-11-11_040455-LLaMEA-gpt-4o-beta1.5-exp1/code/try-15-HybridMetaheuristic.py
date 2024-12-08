import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.cr = 0.9  # Crossover probability
        self.f = 0.8   # Differential weight
        self.local_search_iter = 5
        self.init_population()
    
    def init_population(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                            (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0
    
    def __call__(self, func):
        while self.evaluations < self.budget:
            self.run_differential_evolution(func)
            self.run_greedy_local_search(func)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]
    
    def run_differential_evolution(self, func):
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
            crossover = np.random.rand(self.dim) < self.cr
            trial = np.where(crossover, mutant, self.population[i])
            trial_fitness = func(trial)
            self.evaluations += 1
            if trial_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = trial, trial_fitness
            if self.evaluations % 5 == 0:  # Adaptive tuning line 1
                diversity = np.mean(np.std(self.population, axis=0))  # Change line 1
                self.f = 0.5 + 0.5 * diversity * (self.evaluations / self.budget)  # Change line 2

    def run_greedy_local_search(self, func):
        for i in range(self.population_size):
            candidate = self.population[i]
            best_candidate = candidate.copy()
            best_fitness = self.fitness[i]
            fitness_variance = np.var(self.fitness) / np.var(self.fitness + 1e-10)  # Changed line
            perturbation_range = 0.2 * (1 - fitness_variance)  # Changed line
            for _ in range(self.local_search_iter):
                perturbed = candidate + np.random.uniform(-perturbation_range, perturbation_range, self.dim)  # Line modified 2
                perturbed = np.clip(perturbed, self.lower_bound, self.upper_bound)
                perturbed_fitness = func(perturbed)
                self.evaluations += 1
                if perturbed_fitness < best_fitness:
                    best_candidate, best_fitness = perturbed, perturbed_fitness
            if best_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = best_candidate, best_fitness