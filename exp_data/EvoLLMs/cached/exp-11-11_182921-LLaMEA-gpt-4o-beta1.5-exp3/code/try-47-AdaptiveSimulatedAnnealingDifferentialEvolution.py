import numpy as np

class AdaptiveSimulatedAnnealingDifferentialEvolution:
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
        self.scaling_factor = 0.5
        self.crossover_rate = 0.9
        self.initial_temperature = 100
        self.cooling_rate = 0.9
        self.temperature = self.initial_temperature

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            new_population = np.copy(self.population)
            for i in range(self.population_size):
                a, b, c = np.random.choice(self.population_size, 3, replace=False)
                mutant_vector = self.population[a] + self.scaling_factor * (self.population[b] - self.population[c])

                # Simulated annealing-inspired mutation
                perturbation = np.random.uniform(-1, 1, self.dim) * self.temperature
                mutant_vector += perturbation

                # Crossover
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, self.population[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                # Selection
                trial_score = func(trial_vector)
                self.func_evaluations += 1
                if trial_score < func(self.population[i]):
                    new_population[i] = trial_vector
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_position = trial_vector

            self.population = new_population

            # Temperature cooling schedule
            self.temperature *= self.cooling_rate

            # Adaptive adjustment of scaling factor
            self.scaling_factor = 0.5 + 0.3 * np.tanh(np.pi * (self.func_evaluations / self.budget))

        return self.best_position