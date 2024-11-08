import numpy as np

class AGMSS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10  # Adjusted base population size
        self.scale_adjustment_factor = 0.85  # Adjusted scale factor
        self.initial_scale = 0.1 * (self.upper_bound - self.lower_bound)  # Adjusted initial scale
        self.success_threshold = 0.3  # Adjusted success threshold

    def __call__(self, func):
        def compute_gradient(solution, func_val):
            gradient = np.zeros(self.dim)
            epsilon = 1e-5
            for j in range(self.dim):
                perturb = np.zeros(self.dim)
                perturb[j] = epsilon
                gradient[j] = (func(solution + perturb) - func_val) / epsilon
            return gradient

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

            gradient = compute_gradient(best_solution, best_value)
            best_solution -= 0.01 * gradient  # Learning rate for gradient

            success_rate = success_count / self.population_size
            if success_rate > self.success_threshold:
                scale *= 1.0 / self.scale_adjustment_factor
                self.population_size = min(int(self.population_size * 1.2), self.budget - evaluations)
            else:
                scale *= self.scale_adjustment_factor
                self.population_size = max(int(self.population_size * 0.8), 1)

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