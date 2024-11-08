import numpy as np

class FusionSwarmOptimization:
    def __init__(self, budget, dim, num_particles=30, alpha=0.5, beta=2.0, gamma=1.0):
        self.budget, self.dim, self.num_particles, self.alpha, self.beta, self.gamma = budget, dim, num_particles, alpha, beta, gamma

    def __call__(self, func):
        population, velocity = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim)), np.zeros((self.num_particles, self.dim))
        evaluate_fitness = lambda pop: np.array([func(individual) for individual in pop])
        update_position_velocity = lambda best_pos: [((velocity[i] := self.alpha * velocity[i] + self.beta * (r1 := np.random.rand()) * (best_pos - population[i]) + self.gamma * (r2 := np.random.rand()) * (population[i] - population.mean(axis=0))) or (population[i] += velocity[i])) for i in range(self.num_particles)]
        
        fitness = evaluate_fitness(population)
        best_global_pos = population[np.argmin(fitness)]
        
        for _ in range(self.budget - self.num_particles):
            update_position_velocity(best_global_pos)
            fitness = evaluate_fitness(population)
            best_global_pos = population[np.argmin(fitness)]
        
        return best_global_pos