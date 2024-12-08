import numpy as np

class EnhancedHybridOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.local_search_intensity = 0.1
        self.beta = 0.85
        self.best_solution = None
        self.best_fitness = float('inf')
        self.eval_count = 0

    def differential_evolution(self, population, fitness):
        F = 0.7
        CR = 0.85
        new_population = np.copy(population)

        for i in range(self.population_size):
            if self.eval_count >= self.budget:
                break

            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = np.clip(population[a] + F * (population[b] - population[c]), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])

            trial_fitness = fitness(trial)
            self.eval_count += 1

            if trial_fitness < fitness(population[i]):
                new_population[i] = trial
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

        return new_population

    def local_search(self, solution, fitness):
        candidate_solution = np.copy(solution)
        candidate_fitness = fitness(candidate_solution)
        
        for _ in range(int(self.local_search_intensity * self.dim)):
            perturbed_solution = np.clip(candidate_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
            perturbed_fitness = fitness(perturbed_solution)
            self.eval_count += 1

            if perturbed_fitness < candidate_fitness:
                candidate_solution = perturbed_solution
                candidate_fitness = perturbed_fitness

                if candidate_fitness < self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best_solution = candidate_solution

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        
        while self.eval_count < self.budget:
            # Apply Differential Evolution
            population = self.differential_evolution(population, func)

            # Apply Local Search on the best solution found so far
            self.local_search(self.best_solution, func)

        return self.best_solution