import numpy as np

class AMSSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10
        self.scale_adjustment_factor = 0.85
        self.initial_scale = 0.1 * (self.upper_bound - self.lower_bound)
        self.success_threshold = 0.2

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(best_solution)
        evaluations = 1
        scale = self.initial_scale

        while evaluations < self.budget:
            population = np.random.uniform(
                best_solution - scale, best_solution + scale, (self.population_size, self.dim)
            )
            population = np.clip(population, self.lower_bound, self.upper_bound)

            values = np.apply_along_axis(func, 1, population)
            evaluations += self.population_size

            success_count = 0
            for i in range(self.population_size):
                if values[i] < best_value:
                    best_value = values[i]
                    best_solution = population[i]
                    success_count += 1

            success_rate = success_count / self.population_size
            if success_rate > self.success_threshold:
                scale *= 1.0 / self.scale_adjustment_factor
            else:
                scale *= self.scale_adjustment_factor

            if evaluations + self.population_size > self.budget:
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