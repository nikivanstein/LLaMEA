import numpy as np

class EnhancedHybridAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(80, self.budget // 10)  # Adjusted population size for balance
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.6  # Slightly increased to enhance exploration
        self.CR_base = 0.85  # Adjusted for a balanced crossover
        self.adaptation_rate = 0.1  # Increased for more dynamic parameter changes
        self.local_search_intensity = 0.2  # Enhanced local search intensity
        self.mutation_prob = 0.6  # Balanced probability for mutation strategy
        self.elite_frac = 0.2  # Fraction of best individuals for elite-biased mutation

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
                # Self-adaptive F and CR with dynamic adjustment
                F = np.clip(self.F_base + self.adaptation_rate * np.random.randn(), 0.4, 0.9)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.randn(), 0.7, 1.0)

                # Elite-biased mutation strategy
                if np.random.rand() < self.mutation_prob:
                    elite_count = max(1, int(self.elite_frac * self.population_size))
                    elite_indices = np.argpartition(fitness, elite_count)[:elite_count]
                    a, b = population[np.random.choice(elite_indices, 2, replace=False)]
                    c = population[np.random.choice(self.population_size)]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                else:
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

            population[np.argmin(fitness)] = best_individual
            fitness[np.argmin(fitness)] = best_fitness

        # Return best found solution
        return best_individual