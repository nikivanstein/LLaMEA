import numpy as np

class HybridDifferentialEvolutionAdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * self.dim
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.CR = 0.9  # Crossover rate
        self.F = 0.8   # Differential weight
        self.mutation_factor_bounds = (0.5, 1.0)

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            new_population = np.zeros_like(self.population)
            elite = np.copy(self.population[np.argmin([func(ind) for ind in self.population])])  # Retain best
            for i in range(self.population_size):
                # Adaptive mutation with dynamic scaling factor
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                F_dynamic = self.F + (np.random.rand() - 0.5) * 0.1
                if np.random.rand() < 0.2:  # Adaptive mutation threshold
                    F_dynamic = np.random.uniform(*self.mutation_factor_bounds)
                
                # Introduce rotation to search direction
                rotation_matrix = np.eye(self.dim)
                theta = np.pi * np.random.rand()  # Random rotation angle
                rotation_matrix[0, 0] = np.cos(theta)
                rotation_matrix[0, 1] = -np.sin(theta)
                rotation_matrix[1, 0] = np.sin(theta)
                rotation_matrix[1, 1] = np.cos(theta)
                rotated_direction = rotation_matrix @ (b - c)
                
                mutant = np.clip(a + F_dynamic * rotated_direction, self.lower_bound, self.upper_bound)
                
                # Dynamic Crossover
                CR_dynamic = self.CR + (0.1 * (self.budget - evaluations) / self.budget)  # Increase CR
                cross_points = np.random.rand(self.dim) < CR_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial
                if trial_fitness < func(self.population[i]):
                    new_population[i] = trial
                else:
                    new_population[i] = self.population[i]

                if evaluations >= self.budget:
                    break

            # Enhanced Metropolis Sampling with adaptive covariance
            if evaluations < self.budget:
                current_cov = np.cov(new_population.T) * (2.4**2 / self.dim + 0.01)
                proposal = np.random.multivariate_normal(self.best_solution, current_cov)
                proposal = np.clip(proposal, self.lower_bound, self.upper_bound)
                proposal_fitness = func(proposal)
                evaluations += 1
                if proposal_fitness < self.best_fitness:
                    self.best_fitness = proposal_fitness
                    self.best_solution = proposal

            if evaluations >= self.budget:
                break

            # Update population with new solutions
            self.population = new_population
            self.population[np.random.randint(self.population_size)] = elite  # Maintain elite

        return self.best_solution, self.best_fitness