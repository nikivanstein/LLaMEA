import numpy as np

class DELF:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.sigma = (np.gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2) / (np.gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1 / 1.5)
        self.u_values = np.random.normal(0, 1, (self.budget, self.dim))
        self.v_values = np.random.normal(0, 1, (self.budget, self.dim))

    def levy_flight(self, beta=1.5):
        s = np.random.normal(0, self.sigma, self.dim)
        step = s / np.abs(self.u_values[self.current_index] ** (1 / beta))
        levy = 0.01 * step * self.v_values[self.current_index]
        return levy

    def __call__(self, func):
        pop_size = 10
        F = 0.5
        CR = 0.9
        
        best_solution = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        best_fitness = func(best_solution)
        
        for self.current_index in range(self.budget):
            new_population = []
            a, b, c = np.random.choice(range(pop_size), (3, pop_size), replace=False)
            mutants = best_solution + F * (best_solution - np.array(new_population)[a]) + self.levy_flight()
            trials = np.clip(mutants, self.bounds[0], self.bounds[1])
            
            crossover_masks = np.random.rand(self.dim) < CR
            new_vectors = np.where(crossover_masks, trials, best_solution)
            
            new_fitnesses = [func(vec) for vec in new_vectors]
            improvements = np.where(new_fitnesses < best_fitness)
            best_solution = new_vectors[improvements]
            best_fitness = new_fitnesses[improvements]
            new_population += new_vectors.tolist()

        return best_solution