import numpy as np

class DiverseAdaptiveDynamicMutationEnhancedEvolutionaryStrategies(EnhancedEvolutionaryStrategies):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        
    def __call__(self, func):
        self.initialize_population()
        for _ in range(self.budget // self.population_size):
            self.mutate_population(func)
            self.evaluate_population(func)
            self.adjust_mutation()
            self.diverse_mutation()
        return self.best_solution
    
    def diverse_mutation(self):
        population_fitness = [self.fitness(ind) for ind in self.population]
        avg_fitness = np.mean(population_fitness)
        std_fitness = np.std(population_fitness)
        for i in range(len(self.population)):
            relative_fitness = (population_fitness[i] - avg_fitness) / std_fitness
            if relative_fitness > 0:
                self.population[i] += np.random.normal(0, self.sigma * relative_fitness, self.dim)
        self.clip_population_bounds()