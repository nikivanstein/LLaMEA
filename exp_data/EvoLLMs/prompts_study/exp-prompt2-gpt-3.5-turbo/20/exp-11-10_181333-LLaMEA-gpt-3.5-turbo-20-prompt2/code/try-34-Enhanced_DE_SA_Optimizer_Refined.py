import numpy as np

class Enhanced_DE_SA_Optimizer_Refined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Revised code snippet - dynamic population size adjustment based on fitness diversity
        def adjust_population_size(population, pop_size):
            fitness_values = [func(x) for x in population]
            fitness_std = np.std(fitness_values)
            diversity_ratio = 1.0 / (1.0 + fitness_std)  # Fitness-based diversity
            new_pop_size = int(pop_size * diversity_ratio)  # Adjust population size
            return new_pop_size if new_pop_size > 2 else 2  # Ensure at least 2 individuals

        pop_size = 10 * self.dim
        CR = 0.9
        F = 0.8
        T0 = 1.0
        alpha = 0.95

        def chaotic_map_mutation(x, pop, F):
            a, b, c = pop[np.random.choice(len(pop), 3, replace=False)]
            F = np.clip(F + np.random.normal(0, 0.2), 0.2, 1.0)  # Adaptive F
            chaotic_map = lambda x: np.sin(3.9 * np.sin(3.9 * np.sin(3.9 * x)))  # Chaotic map function
            chaotic_vals = chaotic_map(np.linspace(0, 1, self.dim))
            mutant = np.where(np.random.uniform(0, 1, self.dim) < chaotic_vals, a + F * (b - c), x)
            return np.clip(mutant, -5.0, 5.0)

        def adaptive_simulated_annealing(x, best_x, T):
            new_x = x + np.random.normal(0, T, size=self.dim)
            new_x = np.clip(new_x, -5.0, 5.0)
            T = np.clip(T * 0.99, 0.1, T0)  # Adaptive T
            if func(new_x) < func(x) or np.random.rand() < np.exp((func(x) - func(new_x)) / T):
                return new_x
            else:
                return x

        population = np.random.uniform(-5.0, 5.0, size=(pop_size, self.dim))
        best_x = population[np.argmin([func(x) for x in population)]
        
        for _ in range(self.budget):
            pop_size = adjust_population_size(population, pop_size)  # Dynamic adjustment
            new_population = []
            T = T0 * alpha ** _
            for x in population:
                trial_x = chaotic_map_mutation(x, population, F)
                trial_x = adaptive_simulated_annealing(trial_x, best_x, T)
                new_population.append(trial_x)
                if func(trial_x) < func(best_x):
                    best_x = trial_x
            population = np.array(new_population)
            
        return best_x