import numpy as np

class EnhancedAMSSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(10, dim + 2)  # Adaptive population size
        self.scale_adjustment_factor = 0.8  # Adjusted scale factor
        self.initial_scale = 0.2 * (self.upper_bound - self.lower_bound)  # Larger initial scale
        self.success_threshold = 0.25  # Adjusted success threshold

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_value = func(best_solution)
        evaluations = 1
        scale = self.initial_scale
        success_history = []  # Track success rate over iterations

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
            success_history.append(success_rate)
            if len(success_history) > 5:
                success_history.pop(0)

            avg_success_rate = np.mean(success_history)
            if avg_success_rate > self.success_threshold:
                scale *= 1.1  # Dynamic scale increase
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