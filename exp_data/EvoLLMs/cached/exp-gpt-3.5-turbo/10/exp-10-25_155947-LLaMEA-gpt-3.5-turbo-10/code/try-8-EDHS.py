import numpy as np

class EDHS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.f_min = 0.2
        self.f_max = 0.8
        self.cr = 0.9
        self.hmcr = 0.7
        self.par_bandwidth = 0.3
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

        def differential_mutation(population, current_idx):
            candidates = [idx for idx in range(self.population_size) if idx != current_idx]
            selected = np.random.choice(candidates, 2, replace=False)
            mutant_vector = population[selected[0]] + self.f_min + (self.f_max - self.f_min) * (population[selected[1]] - population[selected[0]])
            return mutant_vector

        population = initialize_population()
        best_solution = population[np.argmin([func(individual) for individual in population])]
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                harmony_memory = population[np.random.randint(0, self.population_size)]
                new_solution = population[i] + self.hmcr * (harmony_memory - population[i])

                if np.random.rand() < self.par_bandwidth:
                    new_solution = differential_mutation(population, i)

                new_solution = np.clip(new_solution, self.lb, self.ub)
                
                if func(new_solution) < func(population[i]):
                    population[i] = new_solution

                    if func(new_solution) < func(best_solution):
                        best_solution = new_solution

        return best_solution