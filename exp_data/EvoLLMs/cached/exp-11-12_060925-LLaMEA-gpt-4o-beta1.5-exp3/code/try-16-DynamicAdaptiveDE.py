import numpy as np

class DynamicAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.initial_population_size = 10 * dim
        self.population = self._initialize_population()
        self.fitness = np.apply_along_axis(self._evaluate, 1, self.population)
        self.global_best_index = np.argmin(self.fitness)
        self.global_best_value = self.fitness[self.global_best_index]
        self.global_best_position = self.population[self.global_best_index]

    def __call__(self, func):
        self.func = func
        while self.evaluations < self.budget:
            for i in range(len(self.population)):
                candidates = [idx for idx in range(len(self.population)) if idx != i]
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
                F = np.random.uniform(0.5, 1.0)  # Scale factor
                trial = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < np.random.uniform(0.1, 0.9)
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, trial, self.population[i])
                f_trial = self._evaluate(trial)

                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
                    if f_trial < self.global_best_value:
                        self.global_best_value = f_trial
                        self.global_best_position = trial
                
                if self.evaluations >= self.budget:
                    break
                
                # Dynamic population adjustment
                if i % (self.dim * 2) == 0:
                    self._adjust_population()

        return self.global_best_position, self.global_best_value

    def _initialize_population(self):
        return self.lower_bound + np.random.rand(self.initial_population_size, self.dim) * (self.upper_bound - self.lower_bound)

    def _evaluate(self, individual):
        fitness = self.func(individual)
        self.evaluations += 1
        return fitness

    def _adjust_population(self):
        # Adjust the population size dynamically based on the evaluation budget
        if self.evaluations > self.budget * 0.5 and len(self.population) > 4 * self.dim:
            # Reduce population size
            self.population = self.population[:4*self.dim]
            self.fitness = self.fitness[:4*self.dim]