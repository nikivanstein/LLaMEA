import numpy as np
import random

class ProbabilisticMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.particle_swarm_size = 20
        self.gene_size = 5
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.particle_swarm = np.random.uniform(-5.0, 5.0, (self.particle_swarm_size, self.dim))
        self.gene_pool = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def __call__(self, func):
        scores = np.zeros(self.population_size)
        for i in range(self.population_size):
            score = func(self.gene_pool[i])
            scores[i] = score

        # Select the best individual
        best_index = np.argmin(scores)
        best_individual = self.gene_pool[best_index]

        # Select the top 20% of individuals with the best scores
        top_individuals = self.gene_pool[np.argsort(scores)[:int(self.population_size * 0.2)]]

        # Update the gene pool
        self.gene_pool = top_individuals

        # Perform particle swarm optimization
        for _ in range(self.budget // 2):
            for i in range(self.particle_swarm_size):
                # Calculate the fitness of the current particle
                score = func(self.particle_swarm[i])

                # Update the particle's position
                self.particle_swarm[i] = self.particle_swarm[i] + np.random.uniform(-1.0, 1.0, self.dim)

                # Update the particle's velocity
                velocity = np.random.uniform(-1.0, 1.0, self.dim)
                self.particle_swarm[i] = self.particle_swarm[i] + velocity

                # Update the particle's score
                score = func(self.particle_swarm[i])

                # Update the best individual
                if score < scores[i]:
                    scores[i] = score
                    best_individual = self.particle_swarm[i]

        # Perform genetic algorithm
        for _ in range(self.budget // 2):
            # Select parents
            parent1 = random.choice(self.gene_pool)
            parent2 = random.choice(self.gene_pool)

            # Perform crossover
            child = parent1 + (parent2 - parent1) * self.crossover_rate

            # Perform mutation
            if random.random() < self.mutation_rate:
                child += np.random.uniform(-1.0, 1.0, self.dim)

            # Update the gene pool
            self.gene_pool = np.concatenate((self.gene_pool, [child]))

            # Update the scores
            score = func(child)
            scores = np.append(scores, score)

        # Update the best individual
        best_individual = self.gene_pool[np.argmin(scores)]

        return best_individual

# Example usage
budget = 100
dim = 5
func = lambda x: x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2
metaheuristic = ProbabilisticMetaheuristic(budget, dim)
best_individual = metaheuristic(func)
print(best_individual)