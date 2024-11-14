import numpy as np

class EAMSSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.base_population_size = 10  # Adjusted population size
        self.scale_adjustment_factor = 0.85  # Adjusted scale factor
        self.initial_scale = 0.2 * (self.upper_bound - self.lower_bound)  # Increased initial scale
        self.success_threshold = 0.3  # Adjusted success threshold

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

            success_count = np.sum(values < best_value)  # Simplified success count
            if success_count > 0:
                best_value = np.min(values)
                best_solution = population[np.argmin(values)]

            success_rate = success_count / population_size
            if success_rate > self.success_threshold:
                scale *= 1.0 / self.scale_adjustment_factor
                population_size = min(int(population_size * 1.2), self.budget - evaluations)  # Adjusted increase rate
            else:
                scale *= self.scale_adjustment_factor
                population_size = max(int(population_size * 0.8), 1)  # Adjusted decrease rate

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
                min_value_idx = np.argmin(values)
                if values[min_value_idx] < best_value:
                    best_value = values[min_value_idx]
                    best_solution = population[min_value_idx]

        return best_solution, best_value