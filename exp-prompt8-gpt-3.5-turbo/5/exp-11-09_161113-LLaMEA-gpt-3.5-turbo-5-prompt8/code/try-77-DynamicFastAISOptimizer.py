def dynamic_mutation_intensity(prev_best_fitness, current_best_fitness, diversity):
    if current_best_fitness < prev_best_fitness:
        return 1.2 * diversity
    else:
        return max(1.0, 0.9 * diversity)

class DynamicFastAISOptimizer(DynamicAISOptimizer):
    def __call__(self, func):
        population_size = self.initial_population_size
        population = initialize_population(population_size)
        diversity = 1.0
        prev_best_fitness = np.inf
        for _ in range(self.max_iterations):
            mutated_population = mutate_population(population, diversity)
            survivors = select_survivors(population, mutated_population, func)
            if len(survivors) == 0:
                break
            best_survivor = elitism_selection(survivors, func)
            population = np.vstack((population, best_survivor))
            population_size = max(1, min(2 * population_size, self.budget // len(population)))
            population = population[:population_size]
            diversity = dynamic_mutation_intensity(prev_best_fitness, np.min(np.apply_along_axis(func, 1, population)), diversity)
            prev_best_fitness = np.min(np.apply_along_axis(func, 1, population))
        best_solution = population[np.argmin(np.apply_along_axis(func, 1, population))]
        return best_solution