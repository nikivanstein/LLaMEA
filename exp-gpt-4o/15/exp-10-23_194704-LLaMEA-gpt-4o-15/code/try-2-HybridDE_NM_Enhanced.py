import numpy as np

class HybridDE_NM_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim
        self.de_cross_over_rate = 0.85  # Slightly adjusted crossover rate for exploration-exploitation balance
        self.de_f = 0.6  # Increased mutation factor for enhanced diversity
        self.simplex_size = dim + 1

    def __call__(self, func):
        budget_used = 0

        # Initialize population with improved diversification
        population = self.lower_bound + (self.upper_bound - self.lower_bound) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in population])
        budget_used += self.population_size

        while budget_used < self.budget:
            # Sort population by fitness
            indices = np.argsort(fitness)
            population = population[indices]
            fitness = fitness[indices]

            # Adaptive Differential Evolution step with dynamic selection
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

            # Nelder-Mead Simplex step with enhanced reflection and expansion scaling
            if budget_used + self.simplex_size <= self.budget:
                centroid = np.mean(population[:self.simplex_size-1], axis=0)
                worst = population[self.simplex_size-1]
                reflection = np.clip(centroid + 1.7 * (centroid - worst), self.lower_bound, self.upper_bound)
                reflection_fitness = func(reflection)
                budget_used += 1

                if reflection_fitness < fitness[self.simplex_size-2]:
                    if reflection_fitness < fitness[0]:  # Check for the best position
                        expansion = np.clip(centroid + 2.7 * (centroid - worst), self.lower_bound, self.upper_bound)
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
                            population[j] = population[0] + 0.5 * (population[j] - population[0])
                            fitness[j] = func(population[j])
                            budget_used += 1
                            if budget_used >= self.budget:
                                break

        # Return the best found solution
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]