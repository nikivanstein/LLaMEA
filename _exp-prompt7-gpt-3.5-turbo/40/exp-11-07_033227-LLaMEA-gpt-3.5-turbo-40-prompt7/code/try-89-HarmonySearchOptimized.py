import numpy as np
import concurrent.futures
import asyncio

class HarmonySearchOptimized:
    def __init__(self, budget, dim, hms=10, hmcr=0.7, par=0.3, bw=0.01):
        self.budget, self.dim, self.hms, self.hmcr, self.par, self.bw = budget, dim, hms, hmcr, par, bw
        self.lower_bound, self.upper_bound = -5.0, 5.0

    def generate_new_harmonies(self):
        return np.where(np.random.rand(self.hms, self.dim) < self.hmcr,
                        np.random.uniform(self.lower_bound, self.upper_bound, (self.hms, self.dim)) +
                        np.random.uniform(-self.bw, self.bw, (self.hms, self.dim)),
                        np.random.uniform(self.lower_bound, self.upper_bound, (self.hms, self.dim)))

    async def evaluate_func(self, func, harmony):
        return func(harmony)

    async def evaluate_harmonies(self, func, harmonies):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await asyncio.gather(*[loop.run_in_executor(executor, self.evaluate_func, func, h) for h in harmonies])

    def __call__(self, func):
        harmonies = self.generate_new_harmonies()
        evaluations = 0

        while evaluations < self.budget:
            loop = asyncio.get_event_loop()
            costs = loop.run_until_complete(self.evaluate_harmonies(func, harmonies))
            evaluations += len(harmonies)

        best_harmony = harmonies[np.argmin(costs)]
        return best_harmony