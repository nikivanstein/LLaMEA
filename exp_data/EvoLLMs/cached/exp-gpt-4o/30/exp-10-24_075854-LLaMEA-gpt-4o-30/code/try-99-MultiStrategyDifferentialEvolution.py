import numpy as np

class MultiStrategyDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5 * dim, 60)  # slightly larger adaptive population size
        self.F = 0.5  # moderate differential weight
        self.CR = 0.85  # lower crossover probability for diversity

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        while evals < self.budget:
            # Differential Evolution with Multi-Strategy
            for i in range(self.population_size):
                if evals >= self.budget:
                    break

                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                
                if np.random.rand() < 0.3:
                    # Strategy 1: Standard DE mutation
                    mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                else:
                    # Strategy 2: Scaled random vector
                    random_vector = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    mutant = np.clip(a + self.F * (random_vector - a), self.lower_bound, self.upper_bound)

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

            # Enhanced Local Search
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx].copy()
            best_fitness = fitness[best_idx]

            for _ in range(5 + int(0.1 * self.dim)):  # adaptive local search based on dimension
                if evals >= self.budget:
                    break

                step_size = np.random.exponential(0.1, self.dim)
                neighbor = best_individual + np.random.normal(0, step_size, self.dim)
                neighbor = np.clip(neighbor, self.lower_bound, self.upper_bound)
                neighbor_fitness = func(neighbor)
                evals += 1

                if neighbor_fitness < best_fitness:
                    best_individual = neighbor
                    best_fitness = neighbor_fitness

            population[best_idx] = best_individual
            fitness[best_idx] = best_fitness

            # Dynamic Parameter Adjustment
            if evals > 0.3 * self.budget:
                self.F = 0.4 + 0.3 * (1 - evals / self.budget)  # gradually reduce F
                self.CR = 0.8 + 0.15 * (evals / self.budget)  # gradually increase CR

        # Return the best solution found
        return population[np.argmin(fitness)]