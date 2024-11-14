import numpy as np

class DifferentialAdaptiveDynamicMutationEnhancedEvolutionaryStrategies(EnhancedEvolutionaryStrategies):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        
    def __call__(self, func):
        self.initialize_population()
        for _ in range(self.budget // self.population_size):
            self.mutate_population(func)
            self.evaluate_population(func)
            self.adjust_mutation()
            self.explore_with_differential_evolution()
        return self.best_solution
    
    def explore_with_differential_evolution(self):
        for i in range(self.population_size):
            candidates = [ind for ind in range(self.population_size) if ind != i]
            random_selection = np.random.choice(candidates, 2, replace=False)
            mutant = self.population[random_selection[0]] + self.sigma * (self.population[random_selection[1]] - self.population[random_selection[2]])
            trial_solution = self.population[i] + self.sigma * (mutant - self.population[i])
            if self.evaluate_fitness(trial_solution) < self.fitness[i]:
                self.population[i] = trial_solution
                self.fitness[i] = self.evaluate_fitness(trial_solution)