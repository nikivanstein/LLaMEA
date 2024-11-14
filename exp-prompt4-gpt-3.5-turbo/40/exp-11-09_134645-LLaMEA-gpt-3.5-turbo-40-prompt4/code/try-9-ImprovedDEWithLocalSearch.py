import numpy as np

class ImprovedDEWithLocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 30
        max_ls_iter = 5

        def target_to_bounds(target, lower, upper):
            return np.clip(target, lower, upper)

        def local_search(candidate, f_local):
            best_candidate = np.copy(candidate)
            for _ in range(max_ls_iter):
                new_candidate = target_to_bounds(best_candidate + 0.01 * np.random.randn(self.dim), -5.0, 5.0)
                if f_local(new_candidate) < f_local(best_candidate):
                    best_candidate = new_candidate
            return best_candidate

        def mutate(current, candidates, F, CR):
            mutated = np.copy(current)
            for i in range(self.dim):
                if np.random.rand() < CR:
                    a, b, c = np.random.choice(pop_size, 3, replace=False)
                    if np.random.rand() < 0.5:
                        F_i = F + 0.1 * np.random.randn()
                    else:
                        F_i = F
                    mutated[i] = target_to_bounds(candidates[a, i] + F_i * (candidates[b, i] - candidates[c, i]), -5.0, 5.0)
                    # Introducing opposition-based learning mutation
                    if np.random.rand() < 0.5:
                        mutated[i] = 2 * 0 - mutated[i]
            return mutated

        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        F = 0.8
        CR = 0.9

        for _ in range(self.budget // pop_size):
            population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
            for i in range(pop_size):
                trial = mutate(population[i], population, F, CR)
                mutated_fitness = func(trial)
                if mutated_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = mutated_fitness
                population[i] = local_search(population[i], func)
        
        return best_solution