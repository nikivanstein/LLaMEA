import numpy as np

class EAMSSA_Refined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.base_population_size = 15  # Increased initial population size
        self.scale_adjustment_factor = 0.85  # Adjusted scale factor
        self.initial_scale = 0.1 * (self.upper_bound - self.lower_bound)  # Refined initial scale
        self.success_threshold = 0.3  # Adjusted success threshold

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(best_solution)
        evaluations = 1
        scale = self.initial_scale
        population_size = self.base_population_size

        while evaluations < self.budget:
            # Generate new population within adaptive scale boundaries
            population = np.random.uniform(
                best_solution - scale, best_solution + scale, (population_size, self.dim)
            )
            population = np.clip(population, self.lower_bound, self.upper_bound)

            # Evaluate the population
            values = np.apply_along_axis(func, 1, population)
            evaluations += population_size

            success_count = 0
            for i in range(population_size):
                if values[i] < best_value:
                    best_value = values[i]
                    best_solution = population[i]
                    success_count += 1

            # Adaptive exploration-exploitation adjustment
            success_rate = success_count / population_size
            if success_rate > self.success_threshold:
                scale *= 1.0 / self.scale_adjustment_factor
                population_size = min(int(population_size * 1.2), self.budget - evaluations)  # Slightly increase pop size
            else:
                scale *= self.scale_adjustment_factor
                population_size = max(int(population_size * 0.8), 1)  # Slightly decrease pop size

            # Ensure evaluations do not exceed budget
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