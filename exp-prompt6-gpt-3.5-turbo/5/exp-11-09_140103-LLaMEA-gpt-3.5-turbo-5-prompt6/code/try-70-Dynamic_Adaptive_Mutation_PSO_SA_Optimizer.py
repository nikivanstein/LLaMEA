# import numpy as np

class Dynamic_Adaptive_Mutation_PSO_SA_Optimizer:
    def __init__(self, budget, dim, num_particles=30, alpha=0.9, beta=2.0, initial_temp=1000.0, final_temp=0.1, temp_decay=0.99, mutation_scale=0.1, dynamic_scale_factor=0.1):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.alpha = alpha
        self.beta = beta
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.temp_decay = temp_decay
        self.mutation_scale = mutation_scale
        self.dynamic_scale_factor = dynamic_scale_factor

    def __call__(self, func):
        def sa_optimize(obj_func, lower_bound, upper_bound, temp, max_iter, mutation_scale, acceptance_prob_history):
            current_state = np.random.uniform(low=lower_bound, high=upper_bound, size=self.dim)
            best_state = current_state
            best_fitness = obj_func(best_state)
            for _ in range(max_iter):
                dynamic_mutation_scale = mutation_scale * np.exp(-self.dynamic_scale_factor)
                candidate_state = current_state + np.random.normal(0, temp * dynamic_mutation_scale, size=self.dim)
                candidate_state = np.clip(candidate_state, lower_bound, upper_bound)
                candidate_fitness = obj_func(candidate_state)
                if candidate_fitness < best_fitness:
                    best_state = candidate_state
                    best_fitness = candidate_fitness
                acceptance_prob = np.exp((best_fitness - candidate_fitness) / temp)
                acceptance_prob_history.append(acceptance_prob)
                if np.random.rand() < acceptance_prob:
                    current_state = candidate_state
                temp *= self.temp_decay * np.mean(acceptance_prob_history)  # Adjust temperature dynamically based on historical acceptance probabilities for enhanced convergence
            return best_state

        best_solution = None
        for _ in range(self.budget):
            acceptance_prob_history = []
            if np.random.rand() < 0.5:
                best_solution = sa_optimize(func, -5.0, 5.0, self.initial_temp, 100, self.mutation_scale, acceptance_prob_history)
                self.initial_temp *= self.temp_decay ** 1.2
                self.dynamic_scale_factor += 0.05  # Introduce dynamic adaptation of mutation scale factor
            else:
                best_solution = sa_optimize(func, -5.0, 5.0, self.initial_temp, 100, self.mutation_scale, acceptance_prob_history)

        return best_solution