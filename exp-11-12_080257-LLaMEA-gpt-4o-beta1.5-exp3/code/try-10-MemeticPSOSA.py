import numpy as np

class MemeticPSOSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)
        self.eval_count = 0
        
        # PSO parameters
        self.inertia = 0.7
        self.cognitive_component = 1.47
        self.social_component = 1.47
        
        # Initialize particles
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.population_size, np.inf)
        
        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def evaluate_population(self, func):
        scores = np.array([func(ind) for ind in self.positions])
        self.eval_count += len(self.positions)
        for i in range(self.population_size):
            if scores[i] < self.personal_best_scores[i]:
                self.personal_best_scores[i] = scores[i]
                self.personal_best_positions[i] = self.positions[i].copy()
            if scores[i] < self.global_best_score:
                self.global_best_score = scores[i]
                self.global_best_position = self.positions[i].copy()
    
    def update_particles(self):
        r1 = np.random.rand(self.population_size, self.dim)
        r2 = np.random.rand(self.population_size, self.dim)
        cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_positions - self.positions)
        social_velocity = self.social_component * r2 * (self.global_best_position - self.positions)
        self.velocities = self.inertia * self.velocities + cognitive_velocity + social_velocity
        self.positions += self.velocities
        self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

    def local_search_sa(self, position, func):
        # Simulated Annealing inspired local search
        temp = 1.0
        cooling_rate = 0.98
        best_pos = position.copy()
        best_score = func(best_pos)
        self.eval_count += 1
        
        while self.eval_count < self.budget and temp > 1e-3:
            candidate = best_pos + np.random.normal(0, 0.1, self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            candidate_score = func(candidate)
            self.eval_count += 1
            
            # Accept if better or with a probability if worse
            if candidate_score < best_score or np.exp((best_score - candidate_score) / temp) > np.random.rand():
                best_pos = candidate
                best_score = candidate_score
            temp *= cooling_rate
        
        return best_pos, best_score

    def __call__(self, func):
        self.evaluate_population(func)
        
        while self.eval_count < self.budget:
            self.update_particles()
            self.evaluate_population(func)
            
            # Apply local search
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                improved_pos, improved_score = self.local_search_sa(self.positions[i], func)
                if improved_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = improved_score
                    self.personal_best_positions[i] = improved_pos
                    if improved_score < self.global_best_score:
                        self.global_best_score = improved_score
                        self.global_best_position = improved_pos
        
        return self.global_best_position