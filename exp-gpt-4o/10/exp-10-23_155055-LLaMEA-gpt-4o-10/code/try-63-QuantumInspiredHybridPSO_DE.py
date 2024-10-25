import numpy as np

class QuantumInspiredHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.population_size = 20
        self.upper_bound = 5.0
        self.lower_bound = -5.0
        self.inertia_weight = 0.65  # Adjusted for better balance
        self.cognitive_constant = 1.4  # Fine-tuned for exploration balance
        self.social_constant = 1.7  # Slightly increased for more social influence
        self.F = np.random.uniform(0.4, 0.85)  # Adaptive scale factor for DE
        self.CR = 0.85  # Adjusted crossover for strategic diversity
        self.func_evals = 0

    def __call__(self, func):
        # Initialize PSO
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        # Quantum-inspired position update
        def quantum_update(position, best_position):
            return np.where(np.random.rand(self.dim) < 0.1, 
                            np.random.uniform(self.lower_bound, self.upper_bound, self.dim), 
                            position + np.random.randn(self.dim) * (best_position - position))

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
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive_constant * r1 * (personal_best_positions[i] - positions[i])
                social_component = self.social_constant * r2 * (global_best_position - positions[i])
                velocities[i] = self.inertia_weight * velocities[i] + cognitive_component + social_component
                quantum_position = quantum_update(positions[i], global_best_position)
                positions[i] = np.where(np.random.rand(self.dim) < 0.1, quantum_position, positions[i] + velocities[i])
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

            # Adaptive Differential Evolution for enhanced exploration
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