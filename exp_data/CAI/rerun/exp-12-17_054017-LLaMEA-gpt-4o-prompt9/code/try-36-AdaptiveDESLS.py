import numpy as np

class AdaptiveDESLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(5 + np.floor(2 * np.log(self.dim)))
        self.mutation_factor = 0.7 + 0.1 * np.random.rand()
        self.crossover_rate = 0.85
        self.best_solution = None
        self.best_fitness = np.inf
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.max_stochastic_iter = 10

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.budget -= self.population_size
        
        dynamic_mutation_factor = self.mutation_factor
        dynamic_crossover_rate = self.crossover_rate
        for _ in range(self.budget):
            fitness_mean = np.mean(fitness)
            dynamic_mutation_factor = self.mutation_factor * (0.5 + 0.5 * np.exp(-np.var(fitness) / fitness_mean))

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + dynamic_mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < dynamic_crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial
                        if self.budget > 0:
                            self._stochastic_search(func, trial)

            dynamic_crossover_rate = 0.8 * (self.best_fitness / fitness_mean)

        return self.best_solution

    def _stochastic_search(self, func, solution):
        current_solution = solution.copy()
        current_fitness = func(current_solution)

        for _ in range(self.max_stochastic_iter):
            for d in range(self.dim):
                if self.budget <= 0:
                    break
                step_size = np.random.uniform(0.01, 0.1) * (self.upper_bound - self.lower_bound)
                neighbor = current_solution.copy()
                neighbor[d] += np.random.normal(0, step_size)
                neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
                neighbor_fitness = func(neighbor)

                if neighbor_fitness < current_fitness:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    if neighbor_fitness < self.best_fitness:
                        self.best_fitness = neighbor_fitness
                        self.best_solution = neighbor

                self.budget -= 1