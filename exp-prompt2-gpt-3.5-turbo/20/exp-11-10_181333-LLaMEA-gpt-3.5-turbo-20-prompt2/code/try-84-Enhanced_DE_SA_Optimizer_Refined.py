import numpy as np

class Enhanced_DE_SA_Optimizer_Refined:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10 * self.dim
        CR = 0.9
        F = 0.8
        T0 = 1.0
        alpha = 0.95

        def crowding_selection(population, population_fitness, n_select):
            distances = np.zeros(len(population))
            for i, ind1 in enumerate(population):
                for j, ind2 in enumerate(population):
                    distances[i] += np.linalg.norm(ind1 - ind2)
            selected_indices = np.argsort(distances)[:n_select]
            return population[selected_indices]

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
        best_x = population[np.argmin([func(x) for x in population])

        for _ in range(self.budget):
            new_population = []
            T = T0 * alpha ** _
            for x in population:
                trial_x = chaotic_map_mutation(x, population, F)
                trial_x = adaptive_simulated_annealing(trial_x, best_x, T)
                new_population.append(trial_x)
                if func(trial_x) < func(best_x):
                    best_x = trial_x
            population = np.array(new_population)
            population = crowding_selection(population, [func(x) for x in population], pop_size)  # Introducing crowding-based selection

        return best_x