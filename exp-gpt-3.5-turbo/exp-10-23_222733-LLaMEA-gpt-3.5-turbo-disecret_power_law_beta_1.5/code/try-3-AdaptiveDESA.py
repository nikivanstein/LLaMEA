import numpy as np

class AdaptiveDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7
        self.initial_temp = 1.0
        self.final_temp = 0.001
        self.alpha = (self.initial_temp - self.final_temp) / budget
        self.current_temp = self.initial_temp
    
    def __call__(self, func):
        def mutate(x, pop, i):
            candidates = [ind for ind in range(self.pop_size) if ind != i]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), -5.0, 5.0)
            return mutant

        def crossover(mutant, target, dim):
            trial = np.copy(target)
            for i in range(dim):
                if np.random.rand() > self.crossover_prob:
                    trial[i] = mutant[i]
            return trial
        
        def acceptance_probability(energy, new_energy, temperature):
            if new_energy < energy:
                return 1.0
            return np.exp((energy - new_energy) / temperature)

        def simulated_annealing(x, func):
            energy = func(x)
            for _ in range(self.budget):
                new_x = mutate(x, population, i)
                new_x = crossover(new_x, x, self.dim)
                new_energy = func(new_x)
                if acceptance_probability(energy, new_energy, self.current_temp) > np.random.rand():
                    x = new_x
                    energy = new_energy
                if new_energy < energy:
                    energy = new_energy
            return x

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        best_solution = population[np.argmin([func(ind) for ind in population])]
        
        for i in range(self.pop_size):
            population[i] = simulated_annealing(population[i], func)
        
        best_solution = population[np.argmin([func(ind) for ind in population])]
        
        return best_solution