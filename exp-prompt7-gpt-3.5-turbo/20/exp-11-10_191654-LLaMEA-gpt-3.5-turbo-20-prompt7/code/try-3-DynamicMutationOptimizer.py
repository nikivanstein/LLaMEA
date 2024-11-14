class DynamicMutationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        # Original algorithm code with 20% modification for dynamic mutation
        mutation_rate = 0.5
        for _ in range(self.budget):
            # Update mutation rate based on population diversity
            diversity = calculate_diversity(population)
            mutation_rate = adapt_mutation_rate(mutation_rate, diversity)
            # Perform optimization with updated mutation rate
            population = mutate_population(population, mutation_rate)
            evaluate_population(population, func)
            best_solution = select_best_solution(population)
        return best_solution