import numpy as np

class EAMSSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.base_population_size = 12
        self.scale_adjustment_factor = 0.9
        self.initial_scale = 0.15 * (self.upper_bound - self.lower_bound)
        self.success_threshold = 0.25

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(best_solution)
        evaluations = 1
        scale = self.initial_scale
        population_size = self.base_population_size

        while evaluations < self.budget:
            population = np.random.uniform(
                best_solution - scale, best_solution + scale, (population_size, self.dim)
            )
            population = np.clip(population, self.lower_bound, self.upper_bound)

            values = np.apply_along_axis(func, 1, population)
            evaluations += population_size

            success_count = 0
            for i in range(population_size):
                if values[i] < best_value:
                    best_value = values[i]
                    best_solution = population[i]
                    success_count += 1

            success_rate = success_count / population_size
            if success_rate > self.success_threshold:
                scale *= 1.0 / self.scale_adjustment_factor
                population_size = min(int(population_size * 1.1), self.budget - evaluations)
            else:
                scale *= self.scale_adjustment_factor
                population_size = max(int(population_size * 0.9), 1)

            if evaluations + population_size > self.budget:
                remaining_evals = self.budget - evaluations
                if remaining_evals <= 0:
                    break
                population = np.random.uniform(
                    best_solution - scale, best_solution + scale, (remaining_evals, self.dim)
                )
                population = np.clip(population, self.lower_bound, self.upper_bound)
                values = np.apply_along_axis(func, 1, population)
                evaluations += remaining_evals
                for i in range(remaining_evals):
                    if values[i] < best_value:
                        best_value = values[i]
                        best_solution = population[i]

        return best_solution, best_value