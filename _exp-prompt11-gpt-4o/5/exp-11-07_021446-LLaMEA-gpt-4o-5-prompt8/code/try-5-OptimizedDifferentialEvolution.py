import numpy as np

class OptimizedDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.population = np.random.uniform(-5, 5, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.F = 0.5
        self.Cr_initial = 0.9
        self.Cr_decay = 0.99
        self.evaluations = 0

    def __call__(self, func):
        self.evaluate_population(func)

        while self.evaluations < self.budget:
            indices = np.arange(self.pop_size)
            np.random.shuffle(indices)
            
            for i in range(0, self.pop_size, 3):  # iterate in chunks for better cache utilization
                if self.evaluations >= self.budget:
                    break

                if i + 2 < self.pop_size:
                    parents = indices[i:i+3]
                else:
                    continue  # skip if we don't have a complete set

                a, b, c = parents
                mutant = self.mutate(a, b, c)
                trial = self.crossover(self.population[i], mutant)

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = trial_fitness

            self.Cr_initial *= self.Cr_decay

        return self.best_solution

    def evaluate_population(self, func):
        fitness_scores = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.pop_size
        self.fitness = fitness_scores
        min_index = np.argmin(fitness_scores)
        self.best_solution = self.population[min_index]
        self.best_fitness = fitness_scores[min_index]

    def mutate(self, a, b, c):
        return np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]), -5, 5)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.Cr_initial
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)