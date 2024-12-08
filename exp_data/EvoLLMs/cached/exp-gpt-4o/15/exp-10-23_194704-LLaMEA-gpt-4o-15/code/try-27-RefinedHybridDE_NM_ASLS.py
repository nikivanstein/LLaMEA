import numpy as np

class RefinedHybridDE_NM_ASLS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim  # Slightly larger population for diversity
        self.de_cross_over_rate = 0.85  # Adjusted crossover rate
        self.de_f = 0.6 + 0.1 * np.random.rand()  # Enhanced stochastic adjustment
        self.simplex_size = dim + 2  # Larger simplex to increase exploration

    def __call__(self, func):
        budget_used = 0

        # Initialize population with diverse and wider spread
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        budget_used += self.population_size

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

            if budget_used >= self.budget:
                break

            # Enhanced Nelder-Mead Simplex step if budget allows
            if budget_used + self.simplex_size <= self.budget:
                centroid = np.mean(population[:self.simplex_size-1], axis=0)
                worst = population[self.simplex_size-1]
                reflection = np.clip(centroid + 1.8 * (centroid - worst), self.lower_bound, self.upper_bound)
                reflection_fitness = func(reflection)
                budget_used += 1

                if reflection_fitness < fitness[self.simplex_size-2]:
                    if reflection_fitness < fitness[0]:  # Check for the best position
                        expansion = np.clip(centroid + 2.8 * (centroid - worst), self.lower_bound, self.upper_bound)
                        expansion_fitness = func(expansion)
                        budget_used += 1

                        if expansion_fitness < reflection_fitness:
                            population[self.simplex_size-1] = expansion
                            fitness[self.simplex_size-1] = expansion_fitness
                        else:
                            population[self.simplex_size-1] = reflection
                            fitness[self.simplex_size-1] = reflection_fitness
                    else:
                        population[self.simplex_size-1] = reflection
                        fitness[self.simplex_size-1] = reflection_fitness
                else:
                    contraction = np.clip(centroid - 0.6 * (centroid - worst), self.lower_bound, self.upper_bound)
                    contraction_fitness = func(contraction)
                    budget_used += 1

                    if contraction_fitness < fitness[self.simplex_size-1]:
                        population[self.simplex_size-1] = contraction
                        fitness[self.simplex_size-1] = contraction_fitness
                    else:
                        for j in range(1, self.simplex_size):
                            population[j] = population[0] + 0.6 * (population[j] - population[0])
                            fitness[j] = func(population[j])
                            budget_used += 1
                            if budget_used >= self.budget:
                                break

            # Adaptive Stochastic Local Search refinement
            for i in range(min(5, self.population_size)):  # Apply to top solutions
                if budget_used >= self.budget:
                    break
                local_perturbation = np.random.normal(0, 0.15, self.dim)
                candidate = np.clip(population[i] + local_perturbation, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                budget_used += 1
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness

        # Return the best found solution
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]