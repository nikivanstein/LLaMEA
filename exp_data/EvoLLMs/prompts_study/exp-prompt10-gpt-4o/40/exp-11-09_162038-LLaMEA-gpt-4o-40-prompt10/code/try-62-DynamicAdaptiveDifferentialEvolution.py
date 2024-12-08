import numpy as np

class DynamicAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(60, self.budget // 6)  # Slightly reduced population size for speed
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.6  # Increased for more aggressive search
        self.CR_base = 0.85  # Slightly decreased to allow more variation
        self.adaptation_rate = 0.1  # Higher rate for quicker adaptation
        self.local_search_intensity = 0.3  # Intensified local search
        self.mutation_prob = 0.8  # Increased probability for more frequent rand-based mutation

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        # Track the best solution found
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]
        
        # Adaptive thresholds
        F_min, F_max = 0.4, 0.9
        CR_min, CR_max = 0.6, 1.0

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Self-adaptive F and CR
                F = np.clip(self.F_base + self.adaptation_rate * np.random.randn(), F_min, F_max)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.randn(), CR_min, CR_max)

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
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            # Enhanced cooperative local search around the best individual
            neighborhood_size = int(self.local_search_intensity * self.population_size)
            local_neighbors = best_individual + np.random.normal(0, 0.02, (neighborhood_size, self.dim))
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