import numpy as np

class EnhancedHybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size_initial = 20 + dim
        self.population_size = self.population_size_initial
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
            
    def __call__(self, func):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = np.random.permutation([idx for idx in range(self.population_size) if idx != i])
                a, b, c = self.population[indices[:3]]
                # Adaptive Mutation Factor (minor tweak)
                self.mutation_factor = 0.6 + 0.3 * (self.evaluations / self.budget)
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                trial_vector = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < self.crossover_probability
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(0, self.dim)] = True
                trial_vector[crossover_points] = mutant_vector[crossover_points]

                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

            if self.evaluations >= self.budget:
                break

            best_idx = np.argmin(self.fitness)
            best_solution = self.population[best_idx]
            covariance_matrix = np.cov(self.population, rowvar=False) * (1 + (self.evaluations / self.budget)) + 1e-5 * np.eye(self.dim)
            mean_solution = np.mean(self.population, axis=0)
            for _ in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                local_solution = np.random.multivariate_normal(mean_solution, covariance_matrix)
                local_solution = np.clip(local_solution, self.lower_bound, self.upper_bound)
                local_fitness = func(local_solution)
                self.evaluations += 1
                if local_fitness < self.fitness[best_idx]:
                    self.population[best_idx] = local_solution
                    self.fitness[best_idx] = local_fitness

            # Periodic Subpopulation Boost
            if self.evaluations % (self.budget // 5) == 0:
                boost_size = int(0.1 * self.population_size)
                boost_population = np.random.uniform(self.lower_bound, self.upper_bound, (boost_size, self.dim))
                for i in range(boost_size):
                    boost_fitness = func(boost_population[i])
                    self.evaluations += 1
                    if boost_fitness < np.max(self.fitness):
                        worst_idx = np.argmax(self.fitness)
                        self.population[worst_idx] = boost_population[i]
                        self.fitness[worst_idx] = boost_fitness

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]