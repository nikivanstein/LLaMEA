import numpy as np

class EnhancedAdaptiveHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 25  # Adjusted for better exploration
        self.population_size = 25
        self.upper_bound = 5.0
        self.lower_bound = -5.0
        self.inertia_weight = 0.6  # Slightly reduced to improve convergence speed
        self.cognitive_constant = 1.9  # Increased for enhanced local search
        self.social_constant = 1.2  # Reduced for stability in convergence
        self.F = np.random.uniform(0.7, 0.9)  # Modified adaptive scale factor
        self.CR = 0.9  # Adjusted crossover for maintaining diversity
        self.func_evals = 0

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        def dynamic_inertia(weight, evals, max_evals):
            return weight * (1 - evals / max_evals) + 0.1  # Added base component for minimal inertia

        while self.func_evals < self.budget:
            for i in range(self.swarm_size):
                if self.func_evals >= self.budget:
                    break
                score = func(positions[i])
                self.func_evals += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            inertia = dynamic_inertia(self.inertia_weight, self.func_evals, self.budget)
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive_constant * r1 * (personal_best_positions[i] - positions[i])
                social_component = self.social_constant * r2 * (global_best_position - positions[i])
                velocities[i] = (inertia * velocities[i] + cognitive_component + social_component)
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

            if self.func_evals + self.population_size * 2 >= self.budget:
                break
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = personal_best_positions[indices]
                mutant_vector = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant_vector, personal_best_positions[i])
                
                if self.func_evals < self.budget:
                    trial_score = func(trial_vector)
                    self.func_evals += 1
                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial_vector
                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial_vector

        return global_best_position, global_best_score