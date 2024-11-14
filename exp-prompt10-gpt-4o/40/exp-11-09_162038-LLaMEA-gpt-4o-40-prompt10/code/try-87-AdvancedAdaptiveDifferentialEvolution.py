import numpy as np

class AdvancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 6)  # Dynamic population size adjustment
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.4  # Fine-tuned for better balance
        self.CR_base = 0.8  # Balanced crossover rate
        self.adaptation_rate = 0.1  # Faster adaptation to changing search dynamics
        self.local_search_intensity = 0.2  # Enhanced local search
        self.mutation_prob = 0.5  # Balanced mutation strategy probability

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
                # Self-adaptive F and CR
                F = np.clip(self.F_base + self.adaptation_rate * np.random.randn(), 0, 1)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.randn(), 0, 1)

                # Multi-phase mutation strategy
                if np.random.rand() < self.mutation_prob:
                    indices = np.random.choice(self.population_size, 4, replace=False)
                    a, b, c, d = population[indices]
                    mutant = np.clip(a + F * (b - c + d - a), self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(best_individual + F * (a - b + c - best_individual), self.lower_bound, self.upper_bound)

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

            # Dynamic local search around the best individual
            if np.random.rand() < self.local_search_intensity:
                neighborhood_size = int(self.local_search_intensity * self.population_size)
                local_neighbors = best_individual + np.random.normal(0, 0.03, (neighborhood_size, self.dim))
                local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
                local_fitness = np.array([func(ind) for ind in local_neighbors])
                eval_count += len(local_neighbors)

                if np.min(local_fitness) < best_fitness:
                    best_local_index = np.argmin(local_fitness)
                    best_individual = local_neighbors[best_local_index]
                    best_fitness = local_fitness[best_local_index]

            population[0] = best_individual
            fitness[0] = best_fitness

        return best_individual