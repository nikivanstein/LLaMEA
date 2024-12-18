import numpy as np

class EnhancedDEAMS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = int(5 + np.floor(2.8 * np.log(self.dim)))
        self.mutation_factor = 0.8 + 0.1 * np.random.rand()
        self.crossover_rate = 0.9
        self.best_solution = None
        self.best_fitness = np.inf
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.max_local_search_iter = 5
        self.successful_mutations = []
        self.success_rate = 0.0
        self.successful_trials = 0
        self.learning_rate = 0.1
        self.diversity_threshold = 0.1
        self.adaptive_learning_rate = self.learning_rate

    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        self.budget -= self.population_size

        dynamic_crossover_rate = self.crossover_rate
        avg_initial_fitness = np.mean(fitness)
        
        for _ in range(self.budget):
            fitness_variance = np.var(fitness)
            fitness_trend = np.mean(fitness) / np.std(fitness)
            population_dispersion = np.mean(np.std(population, axis=0))

            if self.successful_mutations:
                avg_successful_mutation = np.mean(self.successful_mutations)
                adjusted_mutation_factor = self.mutation_factor * (1 + self.adaptive_learning_rate * avg_successful_mutation)
            else:
                adjusted_mutation_factor = self.mutation_factor * (1 + self.adaptive_learning_rate * np.tanh(fitness_variance - fitness_trend))
            
            adjusted_mutation_factor *= (1 + 0.05 * self.success_rate)

            if population_dispersion < self.diversity_threshold:
                adjusted_mutation_factor *= 1.2

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
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
                    self.successful_mutations.append(adjusted_mutation_factor)
                    self.successful_trials += 1
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial
                        if self.budget > 0:
                            self._local_search(func, trial)

            self.success_rate = self.successful_trials / (self.population_size * self.budget)
            dynamic_crossover_rate = 0.9 * (self.best_fitness / np.mean(fitness)) * (1 + 0.1 * self.success_rate)
            avg_current_fitness = np.mean(fitness)
            self.adaptive_learning_rate = self.learning_rate * (1 + np.sign(avg_initial_fitness - avg_current_fitness))

        return self.best_solution

    def _local_search(self, func, solution):
        current_solution = solution.copy()
        current_fitness = func(current_solution)

        for _ in range(self.max_local_search_iter):
            for d in range(self.dim):
                if self.budget <= 0:
                    break
                adaptive_step_size = (self.upper_bound - self.lower_bound) * (0.05 + 0.05 * np.random.rand()) * (1 + self.success_rate)
                neighbor = current_solution.copy()
                neighbor[d] += np.random.uniform(-adaptive_step_size, adaptive_step_size)
                neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
                neighbor_fitness = func(neighbor)

                if neighbor_fitness < current_fitness:
                    current_solution = neighbor
                    current_fitness = neighbor_fitness
                    if neighbor_fitness < self.best_fitness:
                        self.best_fitness = neighbor_fitness
                        self.best_solution = neighbor

                self.budget -= 1