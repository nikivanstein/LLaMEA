import numpy as np

class AdaptiveFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 * dim
        self.alpha = 0.2
        self.beta_min = 0.2
        self.beta_max = 1.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _get_fitness(self, population, func):
        return np.array([func(individual) for individual in population])

    def _update_brightness(self, fitness):
        return self.beta_min + (self.beta_max - self.beta_min) * (1 - fitness)

    def _move_firefly(self, firefly, target, brightness):
        return firefly + brightness * (target - firefly) + self.alpha * np.random.uniform(-1, 1, self.dim)

    def __call__(self, func):
        population = self._initialize_population()
        evals = 0

        while evals < self.budget:
            for i in range(self.population_size):
                target = population[i]
                for j in range(self.population_size):
                    if func(population[j]) < func(target):
                        brightness = self._update_brightness(func(population[j]))
                        target = self._move_firefly(target, population[j], brightness)
                
                population[i] = np.clip(target, self.lower_bound, self.upper_bound)
                evals += 1
                if evals >= self.budget:
                    break

        best_solution = population[np.argmin(self._get_fitness(population, func))]
        return best_solution