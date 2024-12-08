import numpy as np

class MultiStrategyHybridADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = min(50, self.budget // 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F_base = 0.7  # Slightly increased differential weight
        self.CR_base = 0.9  # Increased crossover probability for exploitation
        self.adaptation_rate = 0.05  # Increased adaptation rate
        self.local_search_intensity = 0.2  # Increased local search intensity
        self.mutation_strategies = ['best', 'rand']  # Utilize both strategies
        self.population_size = self.initial_population_size

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
                F = np.clip(self.F_base + self.adaptation_rate * np.random.randn(), 0, 1)
                CR = np.clip(self.CR_base + self.adaptation_rate * np.random.randn(), 0, 1)

                # Dynamic mutation strategy selection
                strategy = np.random.choice(self.mutation_strategies)
                if strategy == 'rand':
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[indices]
                    mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                else:
                    # 'best' strategy
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

            # Enhanced local search using the top individuals
            top_size = max(1, self.population_size // 5)  # Top 20% for local search
            top_indices = np.argsort(fitness)[:top_size]
            for idx in top_indices:
                local_neighbors = population[idx] + np.random.uniform(-0.1, 0.1, (5, self.dim))
                local_neighbors = np.clip(local_neighbors, self.lower_bound, self.upper_bound)
                local_fitness = np.array([func(ind) for ind in local_neighbors])
                eval_count += len(local_neighbors)

                # Update if a better local neighbor is found
                best_local_index = np.argmin(local_fitness)
                if local_fitness[best_local_index] < fitness[idx]:
                    population[idx] = local_neighbors[best_local_index]
                    fitness[idx] = local_fitness[best_local_index]
                    if local_fitness[best_local_index] < best_fitness:
                        best_individual = local_neighbors[best_local_index]
                        best_fitness = local_fitness[best_local_index]

            # Adaptive population size adjustment
            if eval_count < self.budget // 2 and np.random.rand() < 0.1:
                self.population_size = min(self.population_size + 1, self.budget // 5)
                new_individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                new_fitness = func(new_individual)
                eval_count += 1
                if new_fitness < best_fitness:
                    best_individual = new_individual
                    best_fitness = new_fitness
                if new_fitness < np.max(fitness):
                    worst_index = np.argmax(fitness)
                    population[worst_index] = new_individual
                    fitness[worst_index] = new_fitness

        # Return best found solution
        return best_individual