import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 25
        self.population_size = 25
        self.upper_bound = 5.0
        self.lower_bound = -5.0
        self.inertia_weight = 0.9
        self.cognitive_constant = 1.5
        self.social_constant = 2.5
        self.F = 0.9  # Differential evolution scale factor
        self.CR = 0.8  # Crossover probability
        self.func_evals = 0

    def chaos_local_search(self, position):
        perturbation = 0.01 * (np.random.rand(self.dim) - 0.5)
        new_position = position + perturbation
        return np.clip(new_position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        # Initialize PSO
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
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
            
            # Adaptive inertia weight
            self.inertia_weight = 0.9 - 0.7 * (self.func_evals / self.budget)
            
            # Update velocities and positions for PSO
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.cognitive_constant * r1 * (personal_best_positions[i] - positions[i])
                social_component = self.social_constant * r2 * (global_best_position - positions[i])
                velocities[i] = (self.inertia_weight * velocities[i] + cognitive_component + social_component)
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)
            
            # Apply Differential Evolution to enhance exploration
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
            
            # Chaotic local search
            if self.func_evals < self.budget:
                chaotic_position = self.chaos_local_search(global_best_position)
                chaotic_score = func(chaotic_position)
                self.func_evals += 1
                if chaotic_score < global_best_score:
                    global_best_score = chaotic_score
                    global_best_position = chaotic_position

        return global_best_position, global_best_score