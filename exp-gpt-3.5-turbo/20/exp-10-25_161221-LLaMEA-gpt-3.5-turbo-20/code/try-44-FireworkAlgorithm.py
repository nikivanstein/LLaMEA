import numpy as np

class FireworkAlgorithm:
    def __init__(self, budget, dim, num_sparks=5, explosion_radius=0.5):
        self.budget = budget
        self.dim = dim
        self.num_sparks = num_sparks
        self.explosion_radius = explosion_radius

    def __call__(self, func):
        def generate_spark(center):
            return np.clip(center + np.random.uniform(-self.explosion_radius, self.explosion_radius, self.dim), -5.0, 5.0)

        center = np.random.uniform(-5.0, 5.0, self.dim)
        best_solution = center
        best_fitness = func(center)

        for _ in range(self.budget):
            sparks = [generate_spark(center) for _ in range(self.num_sparks)]
            spark_fitness = [func(spark) for spark in sparks]
            best_spark_index = np.argmin(spark_fitness)
            if spark_fitness[best_spark_index] < best_fitness:
                best_solution = sparks[best_spark_index]
                best_fitness = spark_fitness[best_spark_index]
            center = best_solution

        return best_solution