import numpy as np

class OptimizedFusionSwarmOptimization:
    def __init__(self, budget, dim, num_particles=30, alpha=0.5, beta=2.0, gamma=1.0):
        self.budget, self.dim, self.num_particles, self.alpha, self.beta, self.gamma = budget, dim, num_particles, alpha, beta, gamma

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))
        velocity = np.zeros((self.num_particles, self.dim))
        evaluate_fitness = lambda pop: np.array([func(ind) for ind in pop])
        update_pos_vel = lambda best_pos, pop, vel: [vel[i] := self.alpha * vel[i] + self.beta * r1 * (best_pos - pop[i]) + self.gamma * r2 * (pop[i] - pop.mean(axis=0)) for i, (r1, r2) in enumerate(np.random.rand(2, self.num_particles))]
        
        for _ in range(self.budget - self.num_particles):
            fitness = evaluate_fitness(population)
            best_pos = population[np.argmin(fitness)]
            update_pos_vel(best_pos, population, velocity)
        
        return population[np.argmin(fitness)]