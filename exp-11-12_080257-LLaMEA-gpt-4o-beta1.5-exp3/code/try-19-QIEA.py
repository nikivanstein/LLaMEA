import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)
        self.eval_count = 0
        self.alpha = 0.99  # Learning rate for rotation gates
        self.q_population = np.random.rand(self.population_size, self.dim, 2)
        self.q_population /= np.linalg.norm(self.q_population, axis=2, keepdims=True)

    def observe(self, q_individual):
        probabilities = q_individual[:, 0]**2
        return np.where(probabilities > np.random.rand(self.dim), self.upper_bound, self.lower_bound)

    def evaluate_population(self, population, func):
        fitness = np.array([func(ind) for ind in population])
        self.eval_count += len(population)
        return fitness

    def update_quantum_population(self, q_individual, best_solution, individual):
        for i in range(self.dim):
            if best_solution[i] != individual[i]:
                theta = -self.alpha if best_solution[i] == self.upper_bound else self.alpha
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                q0, q1 = q_individual[i]
                q_individual[i] = [cos_theta * q0 - sin_theta * q1, sin_theta * q0 + cos_theta * q1]
        q_individual /= np.linalg.norm(q_individual, axis=1, keepdims=True)

    def __call__(self, func):
        population = np.array([self.observe(q_ind) for q_ind in self.q_population])
        fitness = self.evaluate_population(population, func)
        
        while self.eval_count < self.budget:
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            
            for i in range(self.population_size):
                if self.eval_count >= self.budget:
                    break
                trial = self.observe(self.q_population[i])
                trial_fitness = func(trial)
                self.eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                self.update_quantum_population(self.q_population[i], best_solution, population[i])
        
        return population[np.argmin(fitness)]