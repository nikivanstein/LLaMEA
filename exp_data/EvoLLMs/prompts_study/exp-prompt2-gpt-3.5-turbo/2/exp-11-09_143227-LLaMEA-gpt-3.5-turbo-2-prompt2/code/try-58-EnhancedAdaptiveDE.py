import numpy as np

class EnhancedAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20, crowding_factor=0.5):
        super().__init__(budget, dim, F, CR, pop_size)
        self.crowding_factor = crowding_factor
    
    def __call__(self, func):
        def crowding_distance(population, fitness):
            distances = np.zeros(len(population))
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if i != j:
                        distances[i] += np.linalg.norm(population[i] - population[j])
            return distances / np.max(distances)  # Normalize distances
            
        def select_crowded(population, fitness, crowding_factor):
            distances = crowding_distance(population, fitness)
            sorted_indices = np.argsort(-distances)  # Sort in descending order
            selected_indices = sorted_indices[:int(self.pop_size * crowding_factor)]
            return population[selected_indices]
        
        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            adapt_F = self.F * (1.0 - _ / self.budget)  # Adapt F over time
            adapt_CR = self.CR + 0.1 * np.sin(0.9 * np.pi * _ / self.budget)  # Adapt CR with sinusoidal function
            new_population = []
            
            crowded_population = select_crowded(population, fitness, self.crowding_factor)
            for i, target in enumerate(crowded_population):
                mutant = mutate(target, crowded_population, adapt_F)
                trial = crossover(target, mutant, adapt_CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)
            
        best_idx = np.argmin(fitness)
        return population[best_idx]