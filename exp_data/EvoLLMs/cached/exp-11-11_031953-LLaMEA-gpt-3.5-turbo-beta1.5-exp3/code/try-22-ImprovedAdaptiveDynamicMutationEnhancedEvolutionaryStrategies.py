import numpy as np

class ImprovedAdaptiveDynamicMutationEnhancedEvolutionaryStrategies(EnhancedEvolutionaryStrategies):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 10  # Initialize with a smaller population size
        
    def __call__(self, func):
        self.initialize_population()
        for _ in range(self.budget // self.population_size):
            self.mutate_population(func)
            self.evaluate_population(func)
            self.adjust_mutation()
            self.adjust_population_size()
        return self.best_solution
    
    def adjust_population_size(self):
        if self.sigma > 0.5:
            self.population_size = min(20, self.population_size + 2)  # Increase population size if mutation is high
        else:
            self.population_size = max(5, self.population_size - 1)  # Decrease population size if mutation is low