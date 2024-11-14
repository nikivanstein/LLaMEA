class AdaptiveMutationAISOptimizer(AISOptimizer):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_scale = 1.0

    def __call__(self, func):
        def mutate_population(population, fitness):
            return population + np.random.normal(0, self.mutation_scale/np.mean(fitness), size=population.shape)
        
        population = initialize_population()
        for _ in range(self.max_iterations):
            mutated_population = mutate_population(population, np.apply_along_axis(func, 1, population))
            survivors = select_survivors(population, mutated_population, func)
            if len(survivors) == 0:
                break
            population = np.vstack((population, survivors))
            self.mutation_scale *= 0.95  # Update mutation scale
        best_solution = population[np.argmin(np.apply_along_axis(func, 1, population))]
        return best_solution