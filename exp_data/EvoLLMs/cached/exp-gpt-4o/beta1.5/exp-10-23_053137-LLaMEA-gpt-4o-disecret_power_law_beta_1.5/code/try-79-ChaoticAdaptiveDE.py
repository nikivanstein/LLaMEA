import numpy as np

class ChaoticAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 5 * self.dim
        self.F = 0.7
        self.CR = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
    
    def __call__(self, func):
        np.random.seed(42)
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        global_best = population[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)

        # Initialize chaotic sequence
        chaotic_sequence = self.initialize_chaotic_sequence(evaluations)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Chaotic DE Mutation with Self-adaptive Strategies
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                F_dynamic = self.F * chaotic_sequence[evaluations % len(chaotic_sequence)]
                mutant = np.clip(x0 + F_dynamic * (x1 - x2) + 0.15 * (global_best - x0), self.lower_bound, self.upper_bound)

                # Self-Adaptive Crossover Probability
                CR_dynamic = self.CR * chaotic_sequence[evaluations % len(chaotic_sequence)]
                crossover_mask = np.random.rand(self.dim) < (CR_dynamic + 0.1 * (1 - evaluations / self.budget))
                trial = np.where(crossover_mask, mutant, population[i])

                # Evaluate Trial
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < global_best_fitness:
                        global_best = trial
                        global_best_fitness = trial_fitness

        return global_best

    def initialize_chaotic_sequence(self, length):
        r = 3.99  # Parameter for logistic map
        x = np.random.rand()
        sequence = []
        for _ in range(length):
            x = r * x * (1 - x)  # Logistic map equation
            sequence.append(x)
        return np.array(sequence)