class AdaptiveMutationDEImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_factor = 0.5

    def __call__(self, func):
        population = initialize_population(self.dim)
        fitness = evaluate_population(population, func)
        diversity = calculate_diversity(population)
        
        for i in range(self.budget):
            target = select_target(population)
            trial_vector = generate_trial_vector(target, population, self.mutation_factor)
            trial_fitness = evaluate_individual(trial_vector, func)
            
            if trial_fitness < fitness[target]:
                population[target] = trial_vector
                fitness[target] = trial_fitness
                diversity = calculate_diversity(population)
                self.mutation_factor = adapt_mutation_factor(diversity)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]