import numpy as np

class EnhancedRefinedFireflyAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 * dim
        self.alpha = 0.2
        self.beta_min = 0.2
        self.gamma = 1.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.step_size = 0.2
        self.elitism_rate = 0.1
        self.de_crossover_rate = 0.9
        self.de_scaling_factor = 0.8

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def _get_fitness(self, population, func):
        return np.array([func(individual) for individual in population])

    def _attractiveness(self, i, j):
        return self.beta_min + (self.alpha - self.beta_min) * np.exp(-self.gamma * np.linalg.norm(i - j))

    def _update_position(self, individual, best_individual):
        new_position = individual + self._attractiveness(best_individual, individual) * (best_individual - individual) + self.step_size * np.random.normal(0, 1, self.dim)
        return np.clip(new_position, self.lower_bound, self.upper_bound)
    
    def _differential_evolution(self, population, best_individual):
        mutated_population = np.zeros_like(population)
        for i in range(self.pop_size):
            candidates = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]
            mutant_vector = population[i] + self.de_scaling_factor * (a - b)
            crossover_mask = np.random.rand(self.dim) < self.de_crossover_rate
            trial_vector = np.where(crossover_mask, mutant_vector, population[i])
            if func(trial_vector) < func(population[i]):
                mutated_population[i] = trial_vector
            else:
                mutated_population[i] = population[i]
        return mutated_population

    def __call__(self, func):
        population = self._initialize_population()
        evals = 0

        while evals < self.budget:
            fitness_values = self._get_fitness(population, func)
            best_individual = population[np.argmin(fitness_values)]

            # DE Mutation
            mutated_population = self._differential_evolution(population, best_individual)

            for i in range(self.pop_size):
                new_position = self._update_position(mutated_population[i], best_individual)
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