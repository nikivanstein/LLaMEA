import numpy as np

class Fast_Adaptive_Mutation_SA_Optimizer:
    def __init__(self, budget, dim, initial_temp=1000.0, final_temp=0.1, temp_decay=0.99, mutation_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.temp_decay = temp_decay
        self.mutation_scale = mutation_scale

    def __call__(self, func):
        def sa_optimize(obj_func, lower_bound, upper_bound, temp, max_iter, mutation_scale):
            current_state = np.random.uniform(low=lower_bound, high=upper_bound, size=self.dim)
            best_state = current_state
            best_fitness = obj_func(best_state)
            for _ in range(max_iter):
                dynamic_mutation_scale = mutation_scale * np.exp(-np.mean(np.abs(best_state - current_state)))
                candidate_state = current_state + np.random.normal(0, temp * dynamic_mutation_scale, size=self.dim)
                candidate_state = np.clip(candidate_state, lower_bound, upper_bound)
                candidate_fitness = obj_func(candidate_state)
                if candidate_fitness < best_fitness:
                    best_state = candidate_state
                    best_fitness = candidate_fitness
                acceptance_prob = np.exp((best_fitness - candidate_fitness) / temp)
                if np.random.rand() < acceptance_prob:
                    current_state = candidate_state
                temp *= self.temp_decay
            return best_state

        best_solution = None
        for _ in range(self.budget):
            best_solution = sa_optimize(func, -5.0, 5.0, self.initial_temp, 100, self.mutation_scale)

        return best_solution