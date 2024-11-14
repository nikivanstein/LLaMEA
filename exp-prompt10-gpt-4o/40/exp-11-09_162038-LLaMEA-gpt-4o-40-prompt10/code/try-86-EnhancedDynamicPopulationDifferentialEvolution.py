import numpy as np

class EnhancedDynamicPopulationDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(60, self.budget // 8)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.6  # Slightly increased to enhance exploration
        self.CR_base = 0.85  # Reduced for tighter convergence
        self.adaptation_rate = 0.07  # More adaptive rate for parameter tuning
        self.local_search_intensity = 0.2  # Further increased for aggressive local search
        self.mutation_prob = 0.5  # Balanced probability between mutation strategies

    def __call__(self, func):
        # Initialize dynamic population
        population_size = self.initial_population_size
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = population_size

        # Track the best solution found
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        while eval_count < self.budget:
            # Dynamic population adaptation
            if eval_count > self.budget // 2:
                population_size = max(20, population_size // 2)
                population = population[:population_size]
                fitness = fitness[:population_size]

            for i in range(population_size):
                # Self-adaptive F and CR
                F = np.clip(self.F_base + self.adaptation_rate * np.random.randn(), 0, 1)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.randn(), 0, 1)

                if np.random.rand() < self.mutation_prob:
                    indices = np.random.choice(population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                else:
                    mutant = np.clip(best_individual + F * (population[(i+1) % population_size] - population[(i+2) % population_size]), self.lower_bound, self.upper_bound)

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
            neighborhood_size = int(self.local_search_intensity * population_size)
            local_neighbors = best_individual + np.random.normal(0, 0.05, (neighborhood_size, self.dim))
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