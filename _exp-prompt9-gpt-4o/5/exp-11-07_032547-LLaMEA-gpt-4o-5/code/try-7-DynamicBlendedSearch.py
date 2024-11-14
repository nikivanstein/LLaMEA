import numpy as np

class DynamicBlendedSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 22  # Slightly increased population size
        self.scale_factor = 0.85   # Adjusted differential evolution scaling factor
        self.crossover_rate = 0.88  # Slightly reduced crossover probability

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutate(self, population, best_idx):
        indices = np.arange(self.population_size)
        np.random.shuffle(indices)
        for idx in range(self.population_size):
            if idx == best_idx:
                continue
            a, b, c = population[indices[idx]], population[indices[(idx+1) % self.population_size]], population[indices[(idx+2) % self.population_size]]
            mutant_vector = a + self.scale_factor * (b - c)
            yield np.clip(mutant_vector, self.lower_bound, self.upper_bound)
    
    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        return np.where(crossover_mask, mutant, target)

    def _local_search(self, candidate):
        # Stochastic local search with a decreased perturbation scale
        return np.clip(candidate + np.random.normal(0, 0.08, self.dim), self.lower_bound, self.upper_bound)

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(individual) for individual in population])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            best_idx = np.argmin(fitness)
            best_candidate = population[best_idx]
            
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            for idx, mutant in enumerate(self._mutate(population, best_idx)):
                trial_vector = self._crossover(population[idx], mutant)
                trial_vector = self._local_search(trial_vector)

                trial_fitness = func(trial_vector)
                eval_count += 1

                if trial_fitness < fitness[idx]:
                    new_population[idx] = trial_vector
                    new_fitness[idx] = trial_fitness
                else:
                    new_population[idx] = population[idx]
                    new_fitness[idx] = fitness[idx]

                if eval_count >= self.budget:
                    break
            
            population, fitness = new_population, new_fitness

        return population[np.argmin(fitness)]