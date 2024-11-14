import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(4 * self.dim, 20)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        best_solution = None
        best_fitness = np.inf
        stagnation_counter = 0
        stagnation_limit = 10
        
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1
            if self.fitness[i] < best_fitness:
                best_fitness = self.fitness[i]
                best_solution = self.population[i]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                F = 0.5 + np.random.rand() * 0.5
                CR = 0.6 + np.random.rand() * 0.4

                indices = np.random.choice([x for x in range(self.population_size) if x != i], 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < CR
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(self.dim)] = True
                trial[crossover_points] = mutant[crossover_points]

                if np.random.rand() < 0.1:  # Introduce turbulence mechanism
                    perturbation = 0.1 * np.random.uniform(-1, 1, self.dim)
                    trial = np.clip(trial + perturbation, self.lower_bound, self.upper_bound)

                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial
                        stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                if stagnation_counter >= stagnation_limit:
                    self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
                    stagnation_counter = 0

                if self.evaluations >= self.budget:
                    break
        return best_solution