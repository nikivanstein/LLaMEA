# import numpy as np
# from concurrent.futures import ThreadPoolExecutor

class ParallelHybridHarmonyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hmcr = 0.7
        self.par = 0.4
        self.bandwidth = 0.01
        self.hm_size = int(0.5 * budget)
        self.num_threads = 4  # Number of threads for parallel evaluation
        self.pso_inertia = 0.5
        self.pso_c1 = 1.5
        self.pso_c2 = 1.5
        self.pso_swarm_size = 10
        self.pso_max_iter = 5

    def evaluate_solution(self, func, solution):
        return func(solution)

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.hm_size, self.dim))

        harmony_memory = initialize_harmony_memory()
        fitness = np.array([func(solution) for solution in harmony_memory])
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for _ in range(self.budget - self.hm_size):
                new_solutions = []
                futures = []
                for _ in range(self.num_threads):
                    new_solution = np.zeros(self.dim)
                    hmcr = max(0.3, self.hmcr - 0.001)  # Dynamic adjustment of HMCR
                    par = min(0.6, self.par + 0.001)  # Dynamic adjustment of PAR
                    for i in range(self.dim):
                        if np.random.rand() < hmcr:
                            new_solution[i] = harmony_memory[np.random.choice(self.hm_size)][i]
                        else:
                            new_solution[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                        new_solution[i] = new_solution[i] + np.random.uniform(-self.bandwidth, self.bandwidth)
                        new_solution[i] = np.clip(new_solution[i], self.lower_bound, self.upper_bound)
                    new_solutions.append(new_solution)
                    futures.append(executor.submit(lambda x: self.evaluate_solution(func, x), new_solution))
                
                # Asynchronously evaluate new solutions
                new_fitness = np.array([future.result() for future in futures])

                for idx, new_solution in enumerate(new_solutions):
                    worst_index = np.argmax(fitness)
                    if new_fitness[idx] < fitness[worst_index]:
                        harmony_memory[worst_index] = new_solution
                        fitness[worst_index] = new_fitness[idx]

                # Particle Swarm Optimization (PSO) for local exploration
                pso_swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.pso_swarm_size, self.dim))
                pbest = pso_swarm.copy()
                pbest_fitness = np.array([func(p) for p in pbest])
                gbest_index = np.argmin(pbest_fitness)
                gbest = pbest[gbest_index]

                for _ in range(self.pso_max_iter):
                    for i in range(self.pso_swarm_size):
                        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                        pso_velocity = self.pso_inertia * pso_swarm[i] + self.pso_c1 * r1 * (pbest[i] - pso_swarm[i]) + self.pso_c2 * r2 * (gbest - pso_swarm[i])
                        pso_swarm[i] = np.clip(pso_swarm[i] + pso_velocity, self.lower_bound, self.upper_bound)
                        pso_fitness = func(pso_swarm[i])
                        if pso_fitness < pbest_fitness[i]:
                            pbest[i] = pso_swarm[i]
                            pbest_fitness[i] = pso_fitness
                            if pbest_fitness[i] < func(gbest):
                                gbest = pbest[i]
                                gbest_index = i

        best_index = np.argmin(fitness)
        return harmony_memory[best_index], fitness[best_index]