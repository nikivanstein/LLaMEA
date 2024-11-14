import numpy as np

class Enhanced_Faster_Adaptive_Mutation_PSO_SA_Optimizer(Faster_Enhanced_Adaptive_Mutation_PSO_SA_Optimizer):
    def __call__(self, func):
        def sa_optimize_adaptive_mut(obj_func, lower_bound, upper_bound, temp, max_iter, acceptance_prob_history):
            current_state = np.random.uniform(low=lower_bound, high=upper_bound, size=self.dim)
            best_state = current_state
            best_fitness = obj_func(best_state)
            for _ in range(max_iter):
                mutation_scale = np.clip(np.abs(np.random.normal(0, 1, size=self.dim)) * self.mutation_scale, 0.01, 0.5)
                candidate_state = current_state + np.random.normal(0, temp * mutation_scale, size=self.dim)
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
                best_solution = sa_optimize_adaptive_mut(func, -5.0, 5.0, self.initial_temp, 100, acceptance_prob_history)
                self.initial_temp *= self.temp_decay ** 1.2
            else:
                best_solution = sa_optimize_adaptive_mut(func, -5.0, 5.0, self.initial_temp, 100, acceptance_prob_history)

        return best_solution