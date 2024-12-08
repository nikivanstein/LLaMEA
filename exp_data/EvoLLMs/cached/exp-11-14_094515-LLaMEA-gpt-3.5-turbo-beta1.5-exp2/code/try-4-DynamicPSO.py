import numpy as np

class DynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.9
        self.cognitive_weight = 2.0
        self.social_weight = 2.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.current_evals = 0

    def __call__(self, func):
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_fitness = np.array([func(individual) for individual in swarm])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = swarm[global_best_idx].copy()
        
        while self.current_evals < self.budget:
            inertia_weight = self.inertia_weight - (self.inertia_weight - 0.4) * self.current_evals / self.budget
            for i in range(self.population_size):
                cognitive_component = self.cognitive_weight * np.random.rand(self.dim) * (personal_best[i] - swarm[i])
                social_component = self.social_weight * np.random.rand(self.dim) * (global_best - swarm[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)
                fitness = func(swarm[i])
                self.current_evals += 1
                
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = swarm[i].copy()
                    personal_best_fitness[i] = fitness
                    
                if fitness < func(global_best):
                    global_best = swarm[i].copy()
                    
            global_best_idx = np.argmin(personal_best_fitness)
            global_best = personal_best[global_best_idx].copy()
            
            if self.current_evals < self.budget * 0.7:
                self.population_size = max(5, int(self.population_size * 0.9))
            else:
                self.population_size = max(5, int(self.population_size * 0.6))
            
            swarm = swarm[:self.population_size]
            velocities = velocities[:self.population_size]
            personal_best = personal_best[:self.population_size]
            personal_best_fitness = personal_best_fitness[:self.population_size]
        
        return global_best