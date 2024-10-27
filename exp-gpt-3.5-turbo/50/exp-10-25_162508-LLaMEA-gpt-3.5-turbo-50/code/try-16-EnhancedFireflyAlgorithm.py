import numpy as np

class EnhancedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 0.2
        self.beta_min = 0.2
        self.gamma = 1.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size_min = 10
        self.pop_size_max = 50

    def _initialize_population(self, pop_size):
        return np.random.uniform(self.lower_bound, self.upper_bound, (pop_size, self.dim))

    def _levy_flight(self, step_size=0.01):
        beta = 1.5
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return step_size * u / np.abs(v) ** (1 / beta)

    def __call__(self, func):
        pop_size = self.pop_size_min
        population = self._initialize_population(pop_size)
        evals = 0

        while evals < self.budget:
            fitness_values = np.array([func(individual) for individual in population])
            best_individual = population[np.argmin(fitness_values)]

            for i in range(pop_size):
                step = self._levy_flight()
                new_position = population[i] + self._attractiveness(best_individual, population[i]) * (
                            best_individual - population[i]) + step
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                population[i] = new_position
                evals += 1

                if evals >= self.budget:
                    break

            if evals % 100 == 0 and pop_size < self.pop_size_max:
                pop_size += 5
                population = np.vstack((population, self._initialize_population(5)))

        best_solution = population[np.argmin(np.array([func(individual) for individual in population]))]
        return best_solution
