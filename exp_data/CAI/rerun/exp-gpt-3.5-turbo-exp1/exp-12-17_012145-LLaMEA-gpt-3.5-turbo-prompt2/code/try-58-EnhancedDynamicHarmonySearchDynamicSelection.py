from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.optimize import dual_annealing
from scipy.optimize import basinhopping
from scipy.optimize import shgo

class EnhancedDynamicHarmonySearchDynamicSelection:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = 10
        self.par_min = 0.1
        self.par_max = 0.9
        self.bandwidth_min = 0.01
        self.bandwidth_max = 0.1
        self.inertia_weight = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.optimizers = [differential_evolution, minimize, dual_annealing, basinhopping, shgo]
        
    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))

        def update_parameters(iteration):
            par = self.par_min + (self.par_max - self.par_min) * (iteration / self.budget)
            bandwidth = self.bandwidth_min + (self.bandwidth_max - self.bandwidth_min) * (iteration / self.budget)
            return par, bandwidth

        def improvise_harmony(harmony_memory, par, bandwidth):
            new_harmony = np.copy(harmony_memory[np.random.randint(0, self.harmony_memory_size)])
            for i in range(self.dim):
                if np.random.uniform() < par:
                    new_harmony[i] += np.random.uniform(-bandwidth, bandwidth)
                    new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)
            return new_harmony

        harmony_memory = initialize_harmony_memory()
        best_solution = None
        best_fitness = np.inf

        for iteration in range(self.budget):
            par, bandwidth = update_parameters(iteration)
            new_harmony = improvise_harmony(harmony_memory, par, bandwidth)

            chosen_optimizer = np.random.choice(self.optimizers)
            optimizer_result = chosen_optimizer(func, bounds=[(-5.0, 5.0)]*self.dim).x

            new_fitness = func(new_harmony)
            optimizer_fitness = func(optimizer_result)

            if new_fitness < best_fitness:
                best_solution = new_harmony
                best_fitness = new_fitness

            if optimizer_fitness < best_fitness:
                best_solution = optimizer_result
                best_fitness = optimizer_fitness

            idx = np.argmax([func(h) for h in harmony_memory])
            if new_fitness < func(harmony_memory[idx]):
                harmony_memory[idx] = new_harmony

        return best_solution