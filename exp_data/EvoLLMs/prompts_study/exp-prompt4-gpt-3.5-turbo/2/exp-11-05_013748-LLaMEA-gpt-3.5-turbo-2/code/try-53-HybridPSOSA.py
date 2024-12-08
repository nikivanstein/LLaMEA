import numpy as np

class HybridPSOSA:
    def __init__(self, budget, dim, swarm_size=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter

    def __call__(self, func):
        def evaluate_population(population):
            return np.array([func(individual) for individual in population])

        def perturb_solution(solution, scale=0.1):
            return solution + scale * np.random.randn(self.dim)

        def pso_search():
            # PSO initialization
            population = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            pbest = population.copy()
            pbest_fitness = evaluate_population(pbest)
            gbest_idx = np.argmin(pbest_fitness)
            gbest = pbest[gbest_idx].copy()

            # PSO main loop
            for _ in range(self.max_iter):
                for i in range(self.swarm_size):
                    velocity = np.random.rand() * (pbest[i] - population[i]) + np.random.rand() * (gbest - population[i])
                    population[i] += velocity
                    population[i] = np.clip(population[i], -5.0, 5.0)
                    
                    fitness = func(population[i])
                    if fitness < pbest_fitness[i]:
                        pbest[i] = population[i].copy()
                        pbest_fitness[i] = fitness
                        if fitness < pbest_fitness[gbest_idx]:
                            gbest_idx = i
                            gbest = pbest[i].copy()
            
            return gbest

        def sa_search(initial_solution):
            current_solution = initial_solution
            current_fitness = func(current_solution)
            best_solution = current_solution
            best_fitness = current_fitness

            temperature = 1.0
            cooling_rate = 0.95

            for _ in range(self.max_iter):
                new_solution = perturb_solution(current_solution)
                new_fitness = func(new_solution)

                if new_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - new_fitness) / temperature):
                    current_solution = new_solution
                    current_fitness = new_fitness

                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness

                temperature *= cooling_rate

            return best_solution

        # Hybrid PSO-SA
        best_solution = pso_search()
        remaining_budget = self.budget - self.max_iter * self.swarm_size
        if remaining_budget > 0:
            best_solution = sa_search(best_solution)

        return best_solution