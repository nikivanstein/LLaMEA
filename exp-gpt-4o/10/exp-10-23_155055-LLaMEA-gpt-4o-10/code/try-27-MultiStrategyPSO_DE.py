import numpy as np

class MultiStrategyPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.population_size = 20
        self.upper_bound = 5.0
        self.lower_bound = -5.0
        self.inertia_weight = 0.9  # Adaptive inertia for dynamic balance
        self.cognitive_constant = 1.7  # Enhanced exploration
        self.social_constant = 1.5  # Balanced convergence control
        self.F = np.random.uniform(0.7, 1.2)  # Wider range for adaptive scale factor
        self.CR = 0.9  # Reduced crossover probability for diverse solutions
        self.func_evals = 0

    def __call__(self, func):
        # Initialize PSO
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        # Dynamic adjustment of inertia weight
        def dynamic_inertia(weight, evals, max_evals):
            return weight * (1 - evals / max_evals)

        # Optimization loop
        while self.func_evals < self.budget:
            # Evaluate current positions
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

            # Update velocities and positions for PSO
            inertia = dynamic_inertia(self.inertia_weight, self.func_evals, self.budget)
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive_constant * r1 * (personal_best_positions[i] - positions[i])
                social_component = self.social_constant * r2 * (global_best_position - positions[i])
                velocities[i] = inertia * velocities[i] + cognitive_component + social_component
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

            # Stochastic resampling for enhanced exploration
            if self.func_evals + self.population_size >= self.budget:
                break
            for i in range(self.population_size):
                if np.random.rand() < 0.1:  # Introduce stochastic resampling
                    trial_vector = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                else:
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