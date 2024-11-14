import numpy as np

class SynergisticHybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(80, self.budget // 6)  # Larger population for greater diversity
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.7  # Enhanced exploration with increased differential weight
        self.CR_base = 0.85  # Balanced crossover rate for diversity and exploitation
        self.adaptation_rate = 0.1  # More aggressive adaptation rate
        self.local_search_intensity = 0.2  # Intensified local search for rapid convergence
        self.mutation_prob = 0.75  # Adjusted mutation probability
        self.elitism_rate = 0.1  # Elitism to retain top performers

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
            next_generation = []

            for i in range(self.population_size):
                # Self-adaptive F and CR
                F = np.clip(self.F_base + self.adaptation_rate * np.random.randn(), 0, 1.5)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.randn(), 0, 1)

                # Choose mutation strategy based on a probability
                if np.random.rand() < self.mutation_prob:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.population_size, 2, replace=False)
                    a, b = population[indices]
                    mutant = np.clip(best_individual + F * (a - b), self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < CR, mutant, population[i])

                # Evaluate trial individual
                trial_fitness = func(trial)
                eval_count += 1

                # Selection and elitism
                if trial_fitness < fitness[i]:
                    next_generation.append((trial, trial_fitness))
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness
                else:
                    next_generation.append((population[i], fitness[i]))

                if eval_count >= self.budget:
                    break

            # Sort new generation based on fitness and apply elitism
            next_generation.sort(key=lambda x: x[1])
            population = np.array([ind for ind, _ in next_generation[:self.population_size]])
            fitness = np.array([fit for _, fit in next_generation[:self.population_size]])

            # Enhanced cooperative local search around the best individual
            neighborhood_size = int(self.local_search_intensity * self.population_size)
            local_neighbors = best_individual + np.random.normal(0, 0.05, (neighborhood_size, self.dim))
            local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
            local_fitness = np.array([func(ind) for ind in local_neighbors])
            eval_count += len(local_neighbors)

            # Update best if any local neighbor is better
            if np.min(local_fitness) < best_fitness:
                best_local_index = np.argmin(local_fitness)
                best_individual = local_neighbors[best_local_index]
                best_fitness = local_fitness[best_local_index]

            # Inject best individual back into population
            population[0] = best_individual
            fitness[0] = best_fitness

        # Return best found solution
        return best_individual