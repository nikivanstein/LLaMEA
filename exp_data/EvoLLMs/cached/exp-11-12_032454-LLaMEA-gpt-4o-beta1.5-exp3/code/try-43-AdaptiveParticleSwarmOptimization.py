import numpy as np

class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(10, int(budget / (20 * dim)))  # heuristic for population size
        self.w = 0.9  # inertia weight
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.velocity_clamp = (-(self.ub - self.lb), self.ub - self.lb)
        
    def opposition_based_learning(self, position):
        return self.lb + self.ub - position
    
    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(*self.velocity_clamp, (self.population_size, self.dim))
        personal_best_position = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in personal_best_position])
        num_evaluations = self.population_size
        
        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_position[global_best_index]
        global_best_fitness = personal_best_fitness[global_best_index]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Update velocities
                inertia = self.w * velocities[i]
                cognitive = self.c1 * np.random.rand(self.dim) * (personal_best_position[i] - population[i])
                social = self.c2 * np.random.rand(self.dim) * (global_best_position - population[i])
                velocities[i] = inertia + cognitive + social
                velocities[i] = np.clip(velocities[i], *self.velocity_clamp)
                
                # Update positions
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lb, self.ub)
                
                # Opposition-Based Learning
                opposite_position = self.opposition_based_learning(population[i])
                opposite_position = np.clip(opposite_position, self.lb, self.ub)
                
                # Evaluate both current and opposite positions
                current_fitness = func(population[i])
                opposite_fitness = func(opposite_position)
                num_evaluations += 2
                
                # Select the better position
                if opposite_fitness < current_fitness:
                    population[i] = opposite_position
                    current_fitness = opposite_fitness
                
                # Update personal best
                if current_fitness < personal_best_fitness[i]:
                    personal_best_position[i] = population[i]
                    personal_best_fitness[i] = current_fitness
                
                # Update global best
                if current_fitness < global_best_fitness:
                    global_best_position = population[i]
                    global_best_fitness = current_fitness
        
        return global_best_position, global_best_fitness