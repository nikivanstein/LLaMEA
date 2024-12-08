import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # A common heuristic is to use 10 times the dimension
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover probability
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate initial population
            for i in range(self.population_size):
                if self.fitness[i] == np.inf:
                    self.fitness[i] = func(self.population[i])
                    self.evaluations += 1

            # Main loop
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Mutation: select three random individuals
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]

                # Differential Mutation
                mutant = np.clip(a + self.F * (b - c), -5, 5)

                # Crossover
                trial = np.copy(self.population[i])
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == jrand:
                        trial[j] = mutant[j]

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                    # Adapt F and CR dynamically based on success
                    self.F = min(1.0, self.F + 0.1)
                    self.CR = min(1.0, self.CR + 0.1)
                else:
                    # Slightly decrease F and CR if not successful
                    self.F = max(0.1, self.F - 0.01)
                    self.CR = max(0.1, self.CR - 0.01)

        # Return the best found solution
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]

# Example on how to use the optimizer:
# optimizer = AdaptiveDifferentialEvolution(budget=10000, dim=10)
# result = optimizer(your_black_box_function)