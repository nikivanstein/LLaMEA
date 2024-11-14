import numpy as np

class ImprovedDEWithDynamicPopSize:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
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
                    a, b, c = np.random.choice(len(candidates), 3, replace=False)
                    if np.random.rand() < 0.5:
                        F_i = F + 0.1 * np.random.randn()
                    else:
                        F_i = F
                    mutated[i] = target_to_bounds(candidates[a, i] + F_i * (candidates[b, i] - candidates[c, i]), -5.0, 5.0)
            return mutated

        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)

        F = 0.8
        CR = 0.9
        pop_size = 30

        for _ in range(self.budget // pop_size):
            population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
            for i in range(pop_size):
                trial = mutate(population[i], population, F, CR)
                mutated_fitness = func(trial)
                if mutated_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = mutated_fitness
                population[i] = local_search(population[i], func)
                
            # Dynamically adjust population size based on diversity
            avg_dist = np.mean([np.linalg.norm(population[j] - population[k]) for j in range(pop_size) for k in range(j + 1, pop_size)])
            pop_size = min(50, max(20, int(30 + 10 * (1 - avg_dist))))  # Adjust population size based on diversity
        
        return best_solution