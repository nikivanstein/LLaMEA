import numpy as np

class DynamicPopulationSizeMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10
        scaling_factors = np.full(pop_size, 0.5)
        mutation_rates = np.full(pop_size, 0.5)

        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        for _ in range(self.budget // pop_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fitness_values = [func(ind) for ind in population]

            best_idx = np.argmin(fitness_values)
            population[best_idx] = best_solution
            fitness_values[best_idx] = best_fitness

            for idx, ind in enumerate(population):
                mutated_solution = ind + scaling_factors[idx] * np.random.normal(0, 1, self.dim)

                fitness = func(mutated_solution)
                if fitness < fitness_values[idx]:
                    population[idx] = mutated_solution
                    fitness_values[idx] = fitness
                    scaling_factors[idx] *= 1.1
                    if np.random.uniform(0, 1) < 0.2:
                        mutation_rates[idx] *= 1.2
                    else:
                        mutation_rates[idx] *= 0.9
                else:
                    scaling_factors[idx] *= 0.9  # Decrease scaling factor for non-improving solutions
                    mutation_rates[idx] *= 0.8  # Decrease mutation rate for non-improving solutions

                if fitness < best_fitness:
                    best_solution = mutated_solution
                    best_fitness = fitness

            # Dynamic population size adaptation based on fitness improvements
            if np.random.uniform(0, 1) < 0.2:
                pop_size = max(5, min(20, pop_size + np.sum(np.array(fitness_values) < best_fitness)))
                scaling_factors = np.full(pop_size, 0.5)
                mutation_rates = np.full(pop_size, 0.5)

        return best_solution