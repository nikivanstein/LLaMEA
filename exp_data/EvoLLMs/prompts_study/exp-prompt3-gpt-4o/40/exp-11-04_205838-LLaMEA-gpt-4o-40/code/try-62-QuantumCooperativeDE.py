import numpy as np

class QuantumCooperativeDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(40, dim * 6)
        self.quantum_coeff = 0.5  # Quantum behavior probability
        self.scale_factor = 0.8  # Differential evolution scale factor
        self.crossover_rate = 0.9  # Crossover rate
        self.eval_count = 0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        self.eval_count += self.population_size

        while self.eval_count < self.budget:
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                # Differential Evolution mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.scale_factor * (x2 - x3), self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Quantum behavior exploration
                if np.random.rand() < self.quantum_coeff:
                    quantum_shift = np.random.uniform(-1, 1, self.dim)
                    trial = np.clip(trial + quantum_shift, self.lower_bound, self.upper_bound)

                # Selection
                trial_fitness = func(trial)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        best_index = np.argmin(fitness)
        return population[best_index]