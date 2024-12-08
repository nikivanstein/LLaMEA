# import numpy as np

class Hybrid_SA_DE_Optimizer:
    def __init__(self, budget, dim, num_particles=30, alpha=0.9, beta=2.0, initial_temp=1000.0, final_temp=0.1, temp_decay=0.99, mutation_scale=0.1, dynamic_scale_factor=0.1, de_scale_factor=0.5):
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
        self.de_scale_factor = de_scale_factor

    def __call__(self, func):
        def de_mutate(current_state, scale):
            candidates = [current_state]
            for _ in range(self.dim):
                candidate = current_state + np.random.uniform(-scale, scale, size=self.dim)
                candidates.append(np.clip(candidate, -5.0, 5.0))
            return candidates

        best_solution = None
        for _ in range(self.budget):
            acceptance_prob_history = []
            current_state = np.random.uniform(low=-5.0, high=5.0, size=self.dim)
            best_state = current_state
            best_fitness = func(best_state)

            for _ in range(100):
                dynamic_mutation_scale = self.mutation_scale * np.exp(-self.dynamic_scale_factor)
                de_candidates = de_mutate(current_state, dynamic_mutation_scale * self.de_scale_factor)
                for candidate_state in de_candidates:
                    candidate_fitness = func(candidate_state)
                    if candidate_fitness < best_fitness:
                        best_state = candidate_state
                        best_fitness = candidate_fitness
                current_state = best_state

            best_solution = best_state

        return best_solution