import numpy as np

class EnhancedHybridDEAMS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = int(4 + np.floor(3 * np.log(self.dim)))
        self.mutation_factor = 0.8 + 0.1 * np.random.rand()
        self.crossover_rate = 0.9
        self.best_solution = None
        self.best_fitness = np.inf
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.max_local_search_iter = 5
        self.population_adapt_factor = 0.1

    def __call__(self, func):
        np.random.seed(42)
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.budget -= population_size
        
        dynamic_crossover_rate = self.crossover_rate
        for _ in range(self.budget):
            fitness_variance = np.var(fitness)
            convergence_rate = np.mean(fitness) / (np.std(fitness) + 1e-8)
            adjusted_mutation_factor = self.mutation_factor * (1 + 0.1 * np.tanh(fitness_variance)) * (1 + 0.05 * np.tanh(convergence_rate))
            
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + adjusted_mutation_factor * (b - c), self.lower_bound, self.upper_bound)
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
                            self._local_search(func, trial)
            
            dynamic_crossover_rate = 0.9 * (self.best_fitness / np.mean(fitness))
            # Adapt population size based on convergence rate
            population_size = max(4, int(self.initial_population_size * (1 - self.population_adapt_factor * np.tanh(convergence_rate))))
            population = np.resize(population, (population_size, self.dim))
            fitness = np.resize(fitness, population_size)

        return self.best_solution

    def _local_search(self, func, solution):
        current_solution = solution.copy()
        current_fitness = func(current_solution)

        for _ in range(self.max_local_search_iter):
            for d in range(self.dim):
                if self.budget <= 0:
                    break
                step_size = (self.upper_bound - self.lower_bound) * (0.05 + 0.05 * np.random.rand())
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