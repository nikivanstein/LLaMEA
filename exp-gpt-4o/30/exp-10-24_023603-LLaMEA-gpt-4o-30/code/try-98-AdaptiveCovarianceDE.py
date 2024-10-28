import numpy as np

class AdaptiveCovarianceDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 10 * dim
        self.population_size = self.initial_population_size
        self.memory_size = 5
        self.cross_prob = 0.9
        self.F = 0.5
        self.epsilon = 0.01
        self.probability = 0.3
        
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.memory = {
            "F": np.full(self.memory_size, self.F),
            "CR": np.full(self.memory_size, self.cross_prob)
        }
        self.memory_index = 0
        self.cov_matrix = np.eye(self.dim)  # Initialize covariance matrix

    def update_memory(self, F, CR):
        self.memory["F"][self.memory_index] = F
        self.memory["CR"][self.memory_index] = CR
        self.memory_index = (self.memory_index + 1) % self.memory_size

    def dynamic_population_adjustment(self, eval_count):
        fraction = eval_count / self.budget
        self.population_size = int(self.initial_population_size * (1 - fraction * 0.3))

    def multi_elite_selection(self):
        elite_size = max(1, int(self.population_size * np.random.uniform(0.05, 0.2)))
        elite_indices = np.argsort(self.fitness)[:elite_size]
        return self.population[elite_indices]

    def update_covariance_matrix(self, successful_individuals):
        if len(successful_individuals) > 0:
            diff = successful_individuals - np.mean(successful_individuals, axis=0)
            self.cov_matrix = np.cov(diff, rowvar=False)

    def __call__(self, func):
        eval_count = 0
        best_fitness = np.inf
        successful_individuals = []

        while eval_count < self.budget:
            self.dynamic_population_adjustment(eval_count)
            elites = self.multi_elite_selection()
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 5, replace=False)
                a, b, c, d, e = self.population[indices]

                if np.random.rand() < self.probability:
                    F = np.random.choice(self.memory["F"]) + np.random.rand() * self.epsilon
                    mutant = np.clip(a + F * (b - c + np.random.multivariate_normal(np.zeros(self.dim), self.cov_matrix)), self.lower_bound, self.upper_bound)
                else:
                    local_best = np.argmin(self.fitness[indices])
                    a = elites[np.random.randint(len(elites))]
                    F = np.random.choice(self.memory["F"]) + np.random.rand() * self.epsilon
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                
                CR = np.random.choice(self.memory["CR"]) + np.random.rand() * self.epsilon
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                trial_fitness = func(trial)
                eval_count += 1

                if eval_count >= self.budget:
                    break

                if trial_fitness < self.fitness[i]:
                    successful_individuals.append(trial)
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        self.update_memory(F, CR)
            
            self.update_covariance_matrix(np.array(successful_individuals))
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]