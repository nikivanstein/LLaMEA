import numpy as np

class HybridQuantumEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(10, int(budget / (5 * dim)))  # heuristic for population size
        self.alpha = 0.5  # quantum crossover weight
        self.beta = 0.8  # differential weight
        self.crossover_rate = 0.7

    def __call__(self, func):
        # Initialize the quantum-inspired population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size

        # Quantum state superposition
        q_population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

        global_best_index = np.argmin(fitness)
        global_best = population[global_best_index]
        global_best_fitness = fitness[global_best_index]

        while num_evaluations < self.budget:
            for i in range(self.population_size):
                # Quantum crossover to enhance diversity
                q_population[i] = self.alpha * population[i] + (1 - self.alpha) * q_population[i]
                q_population[i] = np.clip(q_population[i], self.lb, self.ub)

            # Differential evolution step
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.beta * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)

                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                trial = np.where(crossover, mutant, q_population[i])
                trial_fitness = func(trial)
                num_evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < global_best_fitness:
                        global_best = trial
                        global_best_fitness = trial_fitness

        return global_best, global_best_fitness