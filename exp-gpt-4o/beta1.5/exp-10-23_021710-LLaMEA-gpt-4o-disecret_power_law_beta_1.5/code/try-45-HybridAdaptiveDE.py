import numpy as np

class HybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.agents = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.best_agent = None
        self.best_fitness = np.inf
        self.F = 0.8  # differential weight
        self.CR = 0.9  # crossover rate
        self.max_evaluations = budget
        self.alpha = 0.95  # cooling factor for F
        self.beta = 0.97  # cooling factor for CR

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.max_evaluations:
            for i in range(self.population_size):
                score = func(self.agents[i])
                evaluations += 1
                
                if score < self.fitness[i]:
                    self.fitness[i] = score

                if score < self.best_fitness:
                    self.best_fitness = score
                    self.best_agent = self.agents[i]

                if evaluations >= self.max_evaluations:
                    break

            for i in range(self.population_size):
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = np.random.choice(indices, 3, replace=False)

                mutant_vector = self.agents[a] + self.F * (self.agents[b] - self.agents[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                trial_vector = np.copy(self.agents[i])
                if np.random.rand() < self.CR:
                    crossover_point = np.random.randint(self.dim)
                    trial_vector[crossover_point] = mutant_vector[crossover_point]

                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < self.fitness[i]:
                    self.agents[i] = trial_vector
                    self.fitness[i] = trial_score

                if trial_score < self.best_fitness:
                    self.best_fitness = trial_score
                    self.best_agent = trial_vector

                if evaluations >= self.max_evaluations:
                    break

            if evaluations < self.max_evaluations:
                for i in range(self.population_size):
                    if np.random.rand() < 0.1:  # exploration chance
                        random_step = np.random.normal(0, 0.1, self.dim)  # small variance for local search
                        candidate_vector = np.clip(self.agents[i] + random_step, self.lower_bound, self.upper_bound)
                        candidate_score = func(candidate_vector)
                        evaluations += 1

                        if candidate_score < self.fitness[i]:
                            self.agents[i] = candidate_vector
                            self.fitness[i] = candidate_score

                        if candidate_score < self.best_fitness:
                            self.best_fitness = candidate_score
                            self.best_agent = candidate_vector

                        if evaluations >= self.max_evaluations:
                            break
            
            self.F *= self.alpha
            self.CR *= self.beta
        
        return self.best_agent, self.best_fitness