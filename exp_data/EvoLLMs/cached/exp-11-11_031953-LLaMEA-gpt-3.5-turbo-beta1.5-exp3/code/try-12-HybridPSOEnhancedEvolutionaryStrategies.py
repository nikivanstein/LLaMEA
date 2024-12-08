import numpy as np

class HybridPSOEnhancedEvolutionaryStrategies(EnhancedEvolutionaryStrategies):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.pso_positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.pso_velocities = np.zeros((self.population_size, self.dim))
    
    def __call__(self, func):
        self.initialize_population()
        for _ in range(self.budget // self.population_size):
            self.update_pso_positions()
            self.mutate_population(func)
            self.evaluate_population(func)
            self.adjust_mutation()
        return self.best_solution
    
    def update_pso_positions(self):
        inertia_weight = 0.5 + 0.5 * (self.budget - self.func_evals) / self.budget
        cognitive_coeff = 0.5
        social_coeff = 0.5
        for i in range(self.population_size):
            self.pso_velocities[i] = inertia_weight * self.pso_velocities[i] \
                + cognitive_coeff * np.random.uniform(0, 1) * (self.personal_best[i] - self.pso_positions[i]) \
                + social_coeff * np.random.uniform(0, 1) * (self.global_best - self.pso_positions[i])
            self.pso_positions[i] = np.clip(self.pso_positions[i] + self.pso_velocities[i], -5.0, 5.0)
        self.population = self.pso_positions