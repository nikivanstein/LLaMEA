import numpy as np

class EnhancedAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20):
        super().__init__(budget, dim, F, CR, pop_size)
    
    def __call__(self, func):
        def chaotic_map(x, a=1.7, b=0.4):
            return np.sin(a * x) * np.cos(b * x)
        
        def mutate(x, population, F):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            chaotic_factor = chaotic_map(np.sum(x))
            return np.clip(a + F * chaotic_factor * (b - c), -5, 5)
        
        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            adapt_F = self.F * (1.0 - _ / self.budget)  # Adapt F over time
            adapt_CR = self.CR + 0.1 * np.sin(0.9 * np.pi * _ / self.budget)  # Adapt CR with sinusoidal function
            new_population = []
            for i, target in enumerate(population):
                mutant = mutate(target, population, adapt_F)
                trial = crossover(target, mutant, adapt_CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]