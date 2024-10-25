import numpy as np

class AdaptiveQuantumPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.population_size = 20
        self.upper_bound = 5.0
        self.lower_bound = -5.0
        self.inertia_weight = 0.7  # Increased slightly for better exploration
        self.cognitive_constant = 1.4  # Adjusted for enhanced individual learning
        self.social_constant = 1.7  # Increased for stronger social pull
        self.F = np.random.uniform(0.4, 0.8)  # Altered scale factor range
        self.CR = 0.8  # Adjusted crossover probability for exploration-exploitation balance
        self.func_evals = 0

    def __call__(self, func):
        # Initialize Quantum-inspired PSO
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        # Quantum rotation operator for velocity adjustment
        def quantum_rotation(v, p_best, g_best, beta):
            random_vector = np.random.uniform(-1, 1, self.dim)
            return v + beta * np.cross(p_best - v, g_best - v) + random_vector

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

            # Update velocities and positions for Quantum-inspired PSO
            beta = np.random.uniform(0, 1.5)  # Quantum step factor
            for i in range(self.swarm_size):
                velocities[i] = quantum_rotation(velocities[i], personal_best_positions[i], global_best_position, beta)
                positions[i] = positions[i] + velocities[i]
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