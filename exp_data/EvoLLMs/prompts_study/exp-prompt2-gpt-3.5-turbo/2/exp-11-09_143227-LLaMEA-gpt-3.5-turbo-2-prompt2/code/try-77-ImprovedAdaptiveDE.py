import numpy as np

class ImprovedAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20, w=0.5, c1=2.0, c2=2.0):
        super().__init__(budget, dim, F, CR, pop_size)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def __call__(self, func):
        def inertia_weight(w, iteration, max_iter):
            return w - (w / max_iter) * iteration

        def mutate(x, population, F):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            return np.clip(a + F * (b - c), -5, 5)

        def crossover(target, trial, CR):
            mask = np.random.rand(self.dim) < CR
            return np.where(mask, trial, target)

        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for i in range(self.budget):
            adapt_F = self.F * (1.0 - i / self.budget)  
            adapt_CR = self.CR + 0.1 * np.sin(0.9 * np.pi * i / self.budget) 
            w = inertia_weight(self.w, i, self.budget)
            new_population = []
            for j, target in enumerate(population):
                mutant = mutate(target, population, adapt_F)
                trial = crossover(target, mutant, adapt_CR)
                new_fitness = func(trial)
                if new_fitness < fitness[j]:
                    population[j] = trial
                    fitness[j] = new_fitness
                new_population.append(population[j])
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]