import numpy as np

class ModifiedHybridDEAMS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(5 + np.floor(2 * np.log(self.dim)))  # Changed population size calculation
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.best_solution = None
        self.best_fitness = np.inf
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.max_local_search_iter = 5
        self.dynamic_adjustment_factor = 0.1  # New parameter for dynamic adjustments

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.budget -= self.population_size

        for _ in range(self.budget):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                # Introduce adaptive mutation factor
                adaptive_mutation = self.mutation_factor * (1 - (i / self.population_size))
                mutant = np.clip(a + adaptive_mutation * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
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
                            self._local_search(func, trial)

        return self.best_solution

    def _local_search(self, func, solution):
        current_solution = solution.copy()
        current_fitness = func(current_solution)

        for _ in range(int(self.max_local_search_iter * (1 + self.dynamic_adjustment_factor))):  # Adjust local search iterations
            for d in range(self.dim):
                if self.budget <= 0:
                    break
                step_size = (self.upper_bound - self.lower_bound) * 0.1 * (1 - self.dynamic_adjustment_factor)  # Adjust step size dynamically
                neighbor = current_solution.copy()
                neighbor[d] += np.random.uniform(-step_size, step_size)
                neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
                neighbor_fitness = func(neighbor)

                if neighbor_fitness < current_fitness:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    if neighbor_fitness < self.best_fitness:
                        self.best_fitness = neighbor_fitness
                        self.best_solution = neighbor

                self.budget -= 1