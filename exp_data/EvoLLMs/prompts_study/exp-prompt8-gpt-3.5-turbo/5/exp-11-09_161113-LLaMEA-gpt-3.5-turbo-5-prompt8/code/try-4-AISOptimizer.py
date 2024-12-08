class AISOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.max_iterations = budget // self.population_size
        self.base_mutation_step = 1.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def mutate_population(population, mutation_step):
            return population + np.random.normal(0, mutation_step, size=population.shape)

        def select_survivors(current_population, mutated_population, func):
            scores_current = np.apply_along_axis(func, 1, current_population)
            scores_mutated = np.apply_along_axis(func, 1, mutated_population)
            return current_population[scores_current < scores_mutated]

        population = initialize_population()
        mutation_step = self.base_mutation_step
        for _ in range(self.max_iterations):
            mutated_population = mutate_population(population, mutation_step)
            survivors = select_survivors(population, mutated_population, func)
            if len(survivors) == 0:
                break
            population = np.vstack((population, survivors))
            # Adapt mutation step based on population performance
            if len(survivors) > 0:
                mutation_step *= np.mean(np.abs(np.apply_along_axis(func, 1, population))) / np.mean(np.abs(np.apply_along_axis(func, 1, mutated_population)))
        best_solution = population[np.argmin(np.apply_along_axis(func, 1, population))]
        return best_solution