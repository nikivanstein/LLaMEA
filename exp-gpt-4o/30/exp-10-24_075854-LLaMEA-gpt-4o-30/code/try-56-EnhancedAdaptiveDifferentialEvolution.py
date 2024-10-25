import numpy as np

class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5 * dim, 40)  # adaptive population size
        self.F = 0.6  # differential weight
        self.CR = 0.9  # crossover probability
        self.elitism_prob = 0.2  # probability of using elitism

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            # Differential Evolution Phase
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                # Mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                # Apply strategic mutation
                F = self.F if np.random.rand() < 0.7 else np.random.uniform(0.4, 0.8)
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                # Adaptive Crossover
                CR = self.CR + 0.1 * np.random.randn()
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Dynamic Local Search Phase with Elitism
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx].copy()
            best_fitness = fitness[best_idx]

            for _ in range(min(3, self.dim)):  # refined local search
                if evals >= self.budget:
                    break

                neighbor = best_individual + np.random.normal(0, 0.1, self.dim)
                neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
                neighbor_fitness = func(neighbor)
                evals += 1

                if neighbor_fitness < best_fitness:
                    best_individual = neighbor
                    best_fitness = neighbor_fitness

            if np.random.rand() < self.elitism_prob:
                population[best_idx] = best_individual
                fitness[best_idx] = best_fitness

            # Adaptive Diversity Control
            if evals < 0.5 * self.budget and len(population) < 2.5 * self.population_size:
                new_individuals = np.random.uniform(self.lower_bound, self.upper_bound, (3, self.dim))
                new_fitness = np.array([func(ind) for ind in new_individuals])
                evals += len(new_individuals)
                population = np.vstack((population, new_individuals))
                fitness = np.hstack((fitness, new_fitness))

        # Return the best solution found
        return population[np.argmin(fitness)]