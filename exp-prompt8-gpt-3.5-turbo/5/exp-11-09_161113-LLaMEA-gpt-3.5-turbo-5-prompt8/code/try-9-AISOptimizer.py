class AISOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.max_iterations = budget // self.population_size
        self.initial_step_size = 1.0

    def __call__(self, func):
        def mutate_population(population, step_size):
            return population + np.random.normal(0, step_size, size=population.shape)

        population = initialize_population()
        step_size = self.initial_step_size
        for _ in range(self.max_iterations):
            mutated_population = mutate_population(population, step_size)
            survivors = select_survivors(population, mutated_population, func)
            if len(survivors) == 0:
                break
            population = np.vstack((population, survivors))
            step_size *= 0.95  # Adaptive step size reduction
        best_solution = population[np.argmin(np.apply_along_axis(func, 1, population))]
        return best_solution