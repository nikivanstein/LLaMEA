import numpy as np

class QuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def quantum_representation(self, individual):
        return np.cos(individual * np.pi / self.upper_bound), np.sin(individual * np.pi / self.upper_bound)

    def quantum_observation(self, q_population):
        angles = np.arctan2(q_population[:, self.dim:], q_population[:, :self.dim])
        return angles * self.upper_bound / np.pi

    def __call__(self, func):
        # Initialize population
        population = self.lower_bound + np.random.rand(self.pop_size, self.dim) * (self.upper_bound - self.lower_bound)
        q_population = np.hstack(self.quantum_representation(population))
        fitness = np.apply_along_axis(func, 1, population)
        self.evaluations = self.pop_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation strategy
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = q_population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.mutation_factor * (b - c)
                
                # Crossover strategy
                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True

                q_trial = np.where(crossover, mutant, q_population[i])
                trial = self.quantum_observation(q_trial.reshape(1, -1)).flatten()
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Evaluate trial individual
                trial_fitness = func(trial)
                self.evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    q_population[i] = q_trial
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update global best
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                # Early stopping if budget is exhausted
                if self.evaluations >= self.budget:
                    break

        return best_individual, best_fitness