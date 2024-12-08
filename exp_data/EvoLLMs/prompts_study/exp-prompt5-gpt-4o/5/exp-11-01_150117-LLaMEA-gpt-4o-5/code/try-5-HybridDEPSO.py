import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.crossover_rate = 0.7
        self.mutation_factor = 0.8
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5

    def __call__(self, func):
        def clip(x):
            return np.clip(x, self.lower_bound, self.upper_bound)
        
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        pop_velocity = np.zeros_like(pop)
        personal_best = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evals = len(personal_best_scores)

        while evals < self.budget:
            # Differential Evolution phase
            for i in range(self.population_size):
                a, b, c = pop[np.random.choice(self.population_size, 3, replace=False)]
                # Adjust mutation factor dynamically
                dynamic_mutation_factor = self.mutation_factor * (1 - evals / self.budget)
                mutant_vector = clip(a + dynamic_mutation_factor * (b - c))
                # Adaptive crossover rate
                adaptive_crossover_rate = self.crossover_rate * (1 - evals / self.budget)
                crossover_mask = np.random.rand(self.dim) < adaptive_crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, pop[i])
                
                trial_score = func(trial_vector)
                evals += 1
                
                if trial_score < personal_best_scores[i]:
                    personal_best[i] = trial_vector
                    personal_best_scores[i] = trial_score
                    if trial_score < global_best_score:
                        global_best = trial_vector
                        global_best_score = trial_score

                if evals >= self.budget:
                    break

            if evals >= self.budget:
                break

            # Particle Swarm Optimization phase
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                pop_velocity[i] = (self.inertia_weight * pop_velocity[i] +
                                   self.cognitive_constant * r1 * (personal_best[i] - pop[i]) +
                                   self.social_constant * r2 * (global_best - pop[i]))
                
                pop[i] = clip(pop[i] + pop_velocity[i])
                score = func(pop[i])
                evals += 1
                
                if score < personal_best_scores[i]:
                    personal_best[i] = pop[i]
                    personal_best_scores[i] = score
                    if score < global_best_score:
                        global_best = pop[i]
                        global_best_score = score

                if evals >= self.budget:
                    break

        return global_best