import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(4 * dim, 40)  # slightly reduced population size for efficiency
        self.F = 0.5  # reduced differential weight
        self.CR = 0.8  # adjusted crossover probability

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

                # Mutation with quantum-inspired approach
                indices = np.random.choice(self.population_size, 5, replace=False)
                a, b, c, d, e = population[indices]
                F_adaptive = np.random.uniform(0.3, 0.7) if np.random.rand() < 0.8 else 0.5
                mutant = np.clip(a + F_adaptive * (b - c) + F_adaptive * (d - e), self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Quantum-Inspired Local Search Phase
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx].copy()
            best_fitness = fitness[best_idx]

            for _ in range(5 + int(0.1 * self.dim)):  # quantum-inspired search intensity
                if evals >= self.budget:
                    break

                neighbor = best_individual + np.random.normal(0, 0.1, self.dim)
                neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
                neighbor_fitness = func(neighbor)
                evals += 1

                if neighbor_fitness < best_fitness:
                    best_individual = neighbor
                    best_fitness = neighbor_fitness

            population[best_idx] = best_individual
            fitness[best_idx] = best_fitness

            # Adaptive Local Entropy Control
            if evals < 0.5 * self.budget and len(population) < 2.0 * self.population_size:
                entropy_level = np.std(population, axis=0)
                if np.mean(entropy_level) < 1.0:
                    new_individuals = np.random.uniform(self.lower_bound, self.upper_bound, (4, self.dim))
                    new_fitness = np.array([func(ind) for ind in new_individuals])
                    evals += len(new_individuals)
                    population = np.vstack((population, new_individuals))
                    fitness = np.hstack((fitness, new_fitness))

        # Return the best solution found
        return population[np.argmin(fitness)]