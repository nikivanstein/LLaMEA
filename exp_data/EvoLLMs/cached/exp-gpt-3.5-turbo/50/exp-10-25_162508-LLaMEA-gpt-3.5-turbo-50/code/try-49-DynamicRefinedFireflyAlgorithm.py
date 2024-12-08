import numpy as np

class DynamicRefinedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 * dim
        self.alpha_min = 0.1
        self.alpha_max = 0.9  # Updated
        self.beta_min = 0.2
        self.beta_max = 0.8  # Updated
        self.gamma = 1.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.step_size_min = 0.1  # Updated
        self.step_size_max = 0.5  # Updated
        self.elitism_rate = 0.1

    def _get_alpha(self, evals):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * evals / self.budget  # Updated

    def _get_beta(self, evals):
        return self.beta_min + (self.beta_max - self.beta_min) * evals / self.budget  # Updated

    def _get_step_size(self, evals):
        return self.step_size_min + (self.step_size_max - self.step_size_min) * evals / self.budget  # Updated

    def _attractiveness(self, i, j, evals):
        return self._get_beta(evals) + (self._get_alpha(evals) - self._get_beta(evals)) * np.exp(-self.gamma * np.linalg.norm(i - j))

    def _update_position(self, individual, best_individual, evals):
        step_size = self._get_step_size(evals)  # Updated
        new_position = individual + self._attractiveness(best_individual, individual, evals) * (best_individual - individual) + step_size * np.random.normal(0, 1, self.dim)
        return np.clip(new_position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        population = self._initialize_population()
        evals = 0

        while evals < self.budget:
            fitness_values = self._get_fitness(population, func)
            best_individual = population[np.argmin(fitness_values)]

            for i in range(self.pop_size):
                new_position = self._update_position(population[i], best_individual, evals)
                population[i] = new_position
                evals += 1

                if evals >= self.budget:
                    break

            # Elitism
            sorted_indices = np.argsort(fitness_values)
            elite_count = int(self.elitism_rate * self.pop_size)
            population[sorted_indices[:elite_count]] = population[np.argmin(fitness_values)]

        best_solution = population[np.argmin(self._get_fitness(population, func))]
        return best_solution