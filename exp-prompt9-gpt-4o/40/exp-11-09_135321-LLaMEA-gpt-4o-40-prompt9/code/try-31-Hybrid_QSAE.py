import numpy as np

class Hybrid_QSAE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Hybrid parameters
        self.num_agents = 60  # Increased agent count for diversity
        self.inertia_weight = 0.6  # More dynamic inertia weight
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.9  # Enhanced learning from best
        self.memory_coeff = 0.5  # New memory component for adaptive learning

        # Differential Evolution parameters
        self.F = 0.9  # Higher scaling factor for diversity
        self.CR = 0.8  # More balanced crossover

        # Initialize positions and velocities
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_agents, self.dim))
        self.velocities = np.random.uniform(-0.3, 0.3, (self.num_agents, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_agents, np.inf)
        self.memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_agents, self.dim))

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def chaotic_map(self, x):
        return 4 * x * (1 - x)  # Logistic map for chaotic exploration

    def levy_flight(self, L):
        return np.random.standard_cauchy(size=L)

    def __call__(self, func):
        evals = 0
        chaos_factor = np.random.rand()
        
        while evals < self.budget:
            # Evaluate agents
            scores = np.apply_along_axis(func, 1, self.positions)
            evals += self.num_agents

            # Update personal and global bests
            for i in range(self.num_agents):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            # Update velocities and positions
            r1, r2 = np.random.rand(self.num_agents, self.dim), np.random.rand(self.num_agents, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
            memory_component = self.memory_coeff * (self.memory - self.positions)
            self.velocities = (self.inertia_weight * self.velocities + cognitive_component + social_component + memory_component) * chaos_factor
            self.positions += self.velocities * np.random.uniform(0.1, 0.5, self.positions.shape)
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # Perform Adaptive Evolution with Lévy flights
            for i in range(self.num_agents):
                indices = [idx for idx in range(self.num_agents) if idx != i]
                x1, x2, x3 = self.positions[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.positions[i])
                
                # Incorporate Levy flights for diverse exploration
                levy_steps = self.levy_flight(self.dim)
                trial_vector += 0.01 * levy_steps * (trial_vector - self.positions[i])
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                
                trial_score = func(trial_vector)

                # Evolution acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score
                    self.memory[i] = trial_vector  # Update memory with successful vectors
            
            chaos_factor = self.chaotic_map(chaos_factor)  # Update chaos factor
            evals += self.num_agents

        return self.global_best_position, self.global_best_score