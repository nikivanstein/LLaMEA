import numpy as np

class AdaptiveHybridDE_NM_PLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim   # Adjusted for better exploration
        self.de_cross_over_rate = 0.85    # Slightly reduced for diversity
        self.de_f = 0.5 + 0.2 * np.random.rand()  # Increased range for perturbation
        self.simplex_size = dim + 2       # Expanded simplex for better exploration

    def __call__(self, func):
        budget_used = 0

        # Initialize population with more diverse spread
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        budget_used += self.population_size

        best_solution = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while budget_used < self.budget:
            # Sort population by fitness
            indices = np.argsort(fitness)
            population = population[indices]
            fitness = fitness[indices]

            # Adaptive Differential Evolution step
            for i in range(self.population_size):
                if budget_used >= self.budget:
                    break
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + self.de_f * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.de_cross_over_rate
                trial = np.where(crossover, mutant, population[i])
                trial_fitness = func(trial)
                budget_used += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            if budget_used >= self.budget:
                break

            # Refined Nelder-Mead Simplex step
            if budget_used + self.simplex_size <= self.budget:
                centroid = np.mean(population[:self.simplex_size-1], axis=0)
                worst = population[self.simplex_size-1]
                reflection = np.clip(centroid + 1.2 * (centroid - worst), self.lower_bound, self.upper_bound)
                reflection_fitness = func(reflection)
                budget_used += 1

                if reflection_fitness < fitness[self.simplex_size-2]:
                    population[self.simplex_size-1] = reflection
                    fitness[self.simplex_size-1] = reflection_fitness
                    if reflection_fitness < best_fitness:
                        best_solution = reflection
                        best_fitness = reflection_fitness
                else:
                    contraction = np.clip(centroid - 0.6 * (centroid - worst), self.lower_bound, self.upper_bound)
                    contraction_fitness = func(contraction)
                    budget_used += 1

                    if contraction_fitness < fitness[self.simplex_size-1]:
                        population[self.simplex_size-1] = contraction
                        fitness[self.simplex_size-1] = contraction_fitness

            # Probabilistic local search refinement
            for i in range(max(3, self.population_size // 5)):  # More focus on top solutions
                if budget_used >= self.budget:
                    break
                local_perturbation = np.random.normal(0, 0.05, self.dim)
                candidate = np.clip(population[i] + local_perturbation, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                budget_used += 1
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < best_fitness:
                        best_solution = candidate
                        best_fitness = candidate_fitness

        # Return the best found solution
        return best_solution, best_fitness