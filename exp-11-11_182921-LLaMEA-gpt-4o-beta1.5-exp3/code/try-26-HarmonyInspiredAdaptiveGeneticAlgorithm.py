import numpy as np

class HarmonyInspiredAdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.best_position = None
        self.harmony_memory_rate = 0.95
        self.adaptation_rate = 0.2

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            new_population = np.copy(self.population)
            for i in range(self.population_size):
                if np.random.rand() < self.harmony_memory_rate:
                    # Pitch adjustment inspired by harmony search
                    harmony_vector = self.population[np.random.choice(self.population_size)]
                    adjust_factor = np.random.uniform(-0.1, 0.1, self.dim)
                    candidate_vector = harmony_vector + adjust_factor * (self.upper_bound - self.lower_bound)
                else:
                    # Crossover inspired by genetic algorithms
                    parents = np.random.choice(self.population_size, 2, replace=False)
                    crossover_point = np.random.randint(1, self.dim)
                    candidate_vector = np.concatenate([
                        self.population[parents[0], :crossover_point],
                        self.population[parents[1], crossover_point:]
                    ])

                candidate_vector = np.clip(candidate_vector, self.lower_bound, self.upper_bound)

                # Selection
                candidate_score = func(candidate_vector)
                self.func_evaluations += 1
                if candidate_score < func(self.population[i]):
                    new_population[i] = candidate_vector
                    if candidate_score < self.best_score:
                        self.best_score = candidate_score
                        self.best_position = candidate_vector
            
            self.population = new_population

            # Adaptive adjustment of harmony memory rate
            self.harmony_memory_rate = 0.95 - 0.3 * (self.func_evaluations / self.budget)

        return self.best_position