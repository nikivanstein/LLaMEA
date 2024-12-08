import numpy as np

class HybridEvoSwarmOptimizer:
    def __init__(self, budget, dim, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lb = -5.0
        self.ub = 5.0
        self.positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.inertia_weight = 0.9  # Increased initial inertia
        self.cognitive_weight = 1.2  # Adaptive cognitive weight
        self.social_weight = 1.8  # Adaptive social weight

    def __call__(self, func):
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate fitness for each particle
            for i in range(self.pop_size):
                fitness = func(self.positions[i])
                evaluations += 1
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]
                if evaluations >= self.budget:
                    break

            # Adaptive weights adjustment
            if evaluations % (self.budget // 10) == 0:
                self.inertia_weight *= 0.95
                self.cognitive_weight += 0.05
                self.social_weight -= 0.05
            
            # Update velocities and positions
            for i in range(self.pop_size):
                self.velocities[i] = (
                    self.inertia_weight * self.velocities[i]
                    + self.cognitive_weight * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
                    + self.social_weight * np.random.rand(self.dim) * (self.global_best_position - self.positions[i])
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

            # Perform evolutionary mutation using tournament selection
            selected_indices = np.random.choice(self.pop_size, size=self.pop_size//2, replace=False)
            for idx in selected_indices:
                candidate1, candidate2 = np.random.choice(self.pop_size, 2, replace=False)
                if self.personal_best_scores[candidate1] < self.personal_best_scores[candidate2]:
                    winner = candidate1
                else:
                    winner = candidate2
                mutation_idx = np.random.randint(0, self.dim)
                mutation_step = np.random.normal(0, 0.1)  # Small mutation
                self.positions[winner][mutation_idx] += mutation_step
                self.positions[winner] = np.clip(self.positions[winner], self.lb, self.ub)

        return self.global_best_position, self.global_best_score