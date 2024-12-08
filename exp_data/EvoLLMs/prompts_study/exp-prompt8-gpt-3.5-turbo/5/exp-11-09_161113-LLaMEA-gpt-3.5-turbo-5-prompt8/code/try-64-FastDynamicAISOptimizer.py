class FastDynamicAISOptimizer(DynamicAISOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_intensity = 1.0

    def __call__(self, func):
        def mutate_population(population, diversity, mutation_intensity):
            return population + np.random.normal(0, 1 + diversity * mutation_intensity, size=population.shape)

        population_size = self.initial_population_size
        population = initialize_population(population_size)
        diversity = 1.0
        for _ in range(self.max_iterations):
            mutated_population = mutate_population(population, diversity, self.mutation_intensity)
            survivors = select_survivors(population, mutated_population, func)
            if len(survivors) == 0:
                break
            best_survivor = elitism_selection(survivors, func)
            population = np.vstack((population, best_survivor))
            population_size = max(1, min(2 * population_size, self.budget // len(population)))
            population = population[:population_size]
            performance_ratio = np.mean(np.apply_along_axis(func, 1, population)) / func(best_survivor)
            self.mutation_intensity *= 1.0 + performance_ratio
            diversity = len(np.unique(population)) / len(population)
        best_solution = population[np.argmin(np.apply_along_axis(func, 1, population))]
        return best_solution