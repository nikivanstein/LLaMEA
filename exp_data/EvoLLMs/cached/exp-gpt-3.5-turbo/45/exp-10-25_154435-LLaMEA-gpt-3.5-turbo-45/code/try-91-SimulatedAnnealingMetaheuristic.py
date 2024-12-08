import numpy as np

class SimulatedAnnealingMetaheuristic:
    def __init__(self, budget, dim, initial_temperature=100, cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def acceptance_probability(self, energy, new_energy, temperature):
        if new_energy < energy:
            return 1.0
        return np.exp((energy - new_energy) / temperature)

    def __call__(self, func):
        current_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_solution = current_solution
        temperature = self.initial_temperature
        for _ in range(self.budget):
            new_solution = current_solution + np.random.normal(0, 1, self.dim)
            energy = func(current_solution)
            new_energy = func(new_solution)
            if self.acceptance_probability(energy, new_energy, temperature) > np.random.rand():
                current_solution = new_solution
            if new_energy < func(best_solution):
                best_solution = new_solution
            temperature *= self.cooling_rate
        return best_solution