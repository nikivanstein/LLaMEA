class DynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 21 if dim <= 10 else 20  # Adjust population size dynamically
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
            inertia_weight = 0.9 - (0.9 - 0.4) * np.exp(-5 * self.current_evals / self.budget)  # Adaptive inertia weight update
            cognitive_weight = self.cognitive_weight - (self.cognitive_weight - 0.5) * self.current_evals / self.budget
            social_weight = self.social_weight - (self.social_weight - 0.5) * self.current_evals / self.budget
            
            for i in range(self.population_size):
                cognitive_component = cognitive_weight * np.random.rand(self.dim) * (personal_best[i] - swarm[i])
                social_component = social_weight * np.random.rand(self.dim) * (global_best - swarm[i])
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
                        global_best_idx = i
                        
            if global_best_idx < self.population_size // 2:
                self.population_size += 1
                swarm = np.append(swarm, np.random.uniform(self.lower_bound, self.upper_bound, (1, self.dim)), axis=0)
                velocities = np.append(velocities, np.zeros((1, self.dim)), axis=0)
                personal_best = np.append(personal_best, swarm[-1:].copy(), axis=0)
                personal_best_fitness = np.append(personal_best_fitness, func(swarm[-1:]))
                
            global_best_idx = np.argmin(personal_best_fitness)
            global_best = personal_best[global_best_idx].copy()
        
        return global_best