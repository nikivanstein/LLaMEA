import numpy as np

class EnhancedMemeticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.8  # Base differential weight
        self.CR_base = 0.9  # Base crossover probability
        self.adaptation_rate = 0.05  # Adaptive change rate for F and CR

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        # Track the best solution found
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Adaptive F and CR
                F = self.F_base + self.adaptation_rate * np.random.randn()
                CR = self.CR_base + self.adaptation_rate * np.random.randn()

                # Mutation and crossover
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = func(trial)
                eval_count += 1

                # Selection and elitism
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            # Local search around the best individual
            neighborhood_size = 5
            local_neighbors = best_individual + np.random.uniform(-0.05, 0.05, (neighborhood_size, self.dim))
            local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
            local_fitness = np.array([func(ind) for ind in local_neighbors])
            eval_count += len(local_neighbors)

            # Update best if any local neighbor is better
            if np.min(local_fitness) < best_fitness:
                best_local_index = np.argmin(local_fitness)
                best_individual = local_neighbors[best_local_index]
                best_fitness = local_fitness[best_local_index]

            population[0] = best_individual
            fitness[0] = best_fitness

        # Return best found solution
        return best_individual