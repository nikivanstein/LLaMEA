import numpy as np

class EnhancedDynamicSearchSpaceExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        best_solution = np.random.uniform(lower_bound, upper_bound, size=self.dim)
        best_fitness = func(best_solution)
        step_size = 0.1 * (upper_bound - lower_bound)  # Adaptive step size
        mutation_rate = 0.1  # Mutation rate for adaptive strategy
        population = [best_solution]  # Initialize population
        for _ in range(self.budget):
            # Introduce Levy flights for exploring new solutions
            levy_step = np.random.standard_cauchy(size=self.dim) / np.sqrt(np.abs(np.random.normal(size=self.dim)))  
            new_solution = best_solution + levy_step * step_size
            new_solution = np.clip(new_solution, lower_bound, upper_bound)
            new_fitness = func(new_solution)
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                step_size *= 0.95  # Self-adaptive strategy enhancement
                # Mutation for exploitation
                if np.random.rand() < mutation_rate:
                    mutation = np.random.normal(scale=0.1, size=self.dim)
                    new_solution = best_solution + mutation
                    new_solution = np.clip(new_solution, lower_bound, upper_bound)
                    new_fitness = func(new_solution)
                    if new_fitness < best_fitness:
                        best_solution = new_solution
                        best_fitness = new_fitness
            # Diversity maintenance through dynamic population update
            if len(population) < 10:  # Maintain population size
                population.append(new_solution)
            else:
                worst_idx = np.argmax([func(ind) for ind in population])
                if new_fitness < func(population[worst_idx]):
                    population[worst_idx] = new_solution
        return best_solution