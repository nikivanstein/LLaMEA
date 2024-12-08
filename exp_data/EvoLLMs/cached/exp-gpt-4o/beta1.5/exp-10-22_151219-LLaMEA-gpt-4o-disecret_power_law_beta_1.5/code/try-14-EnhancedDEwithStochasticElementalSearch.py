import numpy as np

class EnhancedDEwithStochasticElementalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Adjusted population size for improved diversity
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.scale_factor = 0.7  # Modified scale factor for balanced exploration-exploitation
        self.crossover_rate = 0.9  # Higher crossover rate for better recombination
        self.evaluations = 0
        self.local_search_rate = 0.2  # Increased probability of local search for finer improvements

    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                trial_vector = self.mutate(i)
                trial_vector = self.crossover(trial_vector, self.population[i])
                if np.random.rand() < self.local_search_rate:
                    trial_vector = self.stochastic_elemental_search(trial_vector, func)
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.fitness[i] = trial_fitness
                    self.population[i] = trial_vector
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = self.population[a] + self.scale_factor * (self.population[b] - self.population[c])
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)

    def crossover(self, mutant_vector, target_vector):
        crossover = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover):
            crossover[np.random.randint(0, self.dim)] = True
        trial_vector = np.where(crossover, mutant_vector, target_vector)
        return trial_vector

    def stochastic_elemental_search(self, vector, func):
        step_size = np.random.normal(0, 0.2, self.dim)  # Introduced stochastic variations for flexibility
        for _ in range(5):  # Iterated elemental steps for potential multiple refinements
            local_vector = vector + step_size
            local_vector = np.clip(local_vector, self.lower_bound, self.upper_bound)
            if func(local_vector) < func(vector):
                vector = local_vector
        return vector