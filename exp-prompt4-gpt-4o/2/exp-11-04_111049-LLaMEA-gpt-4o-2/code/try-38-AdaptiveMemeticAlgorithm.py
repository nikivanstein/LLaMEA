import numpy as np

class AdaptiveMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.F = 0.8  # Differential evolution mutation factor
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = None

    def evaluate_population(self, func):
        if self.fitness is None:
            self.fitness = np.array([func(ind) for ind in self.population])

    def differential_evolution(self, idx):
        candidates = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        best_fitness = self.fitness.min()
        self.F = np.random.uniform(0.5, 1.0) * (1 - self.current_iteration / self.budget) * (best_fitness / (best_fitness + 1e-9))  # Adjusted mutation factor
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def crossover(self, target, mutant):
        self.CR = 0.5 + 0.5 * (1 - self.current_iteration / self.budget)  # Adaptive crossover probability
        crossover_mask = np.random.rand(self.dim) < self.CR
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        return np.where(crossover_mask, mutant, target)

    def local_search(self, individual, func):
        current_fitness = func(individual)
        perturbation_size = 0.1 * (1 - self.current_iteration / self.budget) * (1 if current_fitness == 0 else current_fitness / self.fitness.min())
        perturbation = np.random.uniform(-perturbation_size, perturbation_size, self.dim)
        candidate = np.clip(individual + perturbation, self.lower_bound, self.upper_bound)
        if func(candidate) < current_fitness:
            return candidate
        return individual

    def __call__(self, func):
        self.evaluate_population(func)
        evaluations = self.population_size

        while evaluations < self.budget:
            self.current_iteration = evaluations
            # Adjust population size adaptively
            self.population_size = max(5, int(20 * (1 - evaluations / self.budget)))  
            for i in range(self.population_size):
                target = self.population[i]
                mutant = self.differential_evolution(i)
                trial = self.crossover(target, mutant)
                trial = self.local_search(trial, func)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]