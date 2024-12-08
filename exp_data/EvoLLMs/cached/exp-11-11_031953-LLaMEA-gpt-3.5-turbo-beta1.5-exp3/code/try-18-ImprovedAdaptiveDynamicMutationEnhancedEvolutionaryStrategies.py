import numpy as np

class ImprovedAdaptiveDynamicMutationEnhancedEvolutionaryStrategies(EnhancedEvolutionaryStrategies):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        
    def __call__(self, func):
        self.initialize_population()
        for _ in range(self.budget // self.population_size):
            self.mutate_population(func)
            self.evaluate_population(func)
            self.adjust_mutation()
        return self.best_solution
    
    def adjust_mutation(self):
        convergence_rate = 1.0 - (self.best_fitness / self.population_size)  # measure convergence rate based on population size
        self.sigma = max(0.1, min(0.9, self.sigma + 0.05 * (convergence_rate - 0.5)))  # adapt mutation strength based on convergence
