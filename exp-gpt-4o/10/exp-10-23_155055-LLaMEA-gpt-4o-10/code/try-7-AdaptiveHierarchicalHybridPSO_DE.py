import numpy as np

class AdaptiveHierarchicalHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 25  # Adjusted for better coverage
        self.population_size = 25
        self.upper_bound = 5.0
        self.lower_bound = -5.0
        self.inertia_weight = 0.7  # Enhanced inertia for better momentum
        self.cognitive_constant = 1.5  # Adjusted cognitive component
        self.social_constant = 1.5  # Balanced social component
        self.F = 0.9  # Higher scale factor for increased diversity
        self.CR = 0.85  # Balanced crossover probability
        self.func_evals = 0

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))  # Initial small random velocities
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        def dynamic_inertia(weight, evals, max_evals):
            return weight * (0.5 + 0.5 * (1 - evals / max_evals))

        def roulette_selection(scores):
            max_score = np.max(scores)
            probabilities = (max_score - scores) / (max_score - np.min(scores) + 1e-10)
            probabilities /= np.sum(probabilities)
            return np.random.choice(self.swarm_size, p=probabilities)

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
                velocities[i] = inertia * velocities[i] + cognitive_component + social_component
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

            if self.func_evals + self.population_size * 2 >= self.budget:
                break
            for i in range(self.population_size):
                idx_best = roulette_selection(personal_best_scores)
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