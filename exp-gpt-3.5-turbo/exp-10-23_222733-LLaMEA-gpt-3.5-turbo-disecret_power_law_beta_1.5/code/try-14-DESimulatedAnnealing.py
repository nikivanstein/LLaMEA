import numpy as np

class DESimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.crossover_rate = 0.9
        self.scale_factor = 0.5
        self.initial_temp = 1.0
        self.final_temp = 0.001
        self.alpha = (self.initial_temp - self.final_temp) / budget
        self.current_temp = self.initial_temp
    
    def __call__(self, func):
        def perturb_position(position):
            perturbed_position = np.clip(position + np.random.uniform(-0.5, 0.5, self.dim), -5.0, 5.0)
            return perturbed_position
        
        def acceptance_probability(energy, new_energy, temperature):
            if new_energy < energy:
                return 1.0
            return np.exp((energy - new_energy) / temperature)
        
        def simulated_annealing(x, func):
            energy = func(x)
            for _ in range(self.budget):
                new_x = perturb_position(x)
                new_energy = func(new_x)
                if acceptance_probability(energy, new_energy, self.current_temp) > np.random.rand():
                    x = new_x
                    energy = new_energy
            return x
        
        def differential_evolution(func):
            population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
            for _ in range(self.budget):
                for i in range(self.pop_size):
                    candidate = population[i]
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = candidate + self.scale_factor * (a - b)
                    trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, candidate)
                    population[i] = simulated_annealing(trial, func)
            return population[np.argmin([func(ind) for ind in population])]
        
        best_solution = differential_evolution(func)
        
        return best_solution