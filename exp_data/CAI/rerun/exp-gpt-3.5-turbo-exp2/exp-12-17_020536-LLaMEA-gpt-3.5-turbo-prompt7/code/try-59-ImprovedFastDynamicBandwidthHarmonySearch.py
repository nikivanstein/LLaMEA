import numpy as np
from joblib import Parallel, delayed
from fast_dynamic_bandwidth_harmony_search import FastDynamicBandwidthHarmonySearch

class ImprovedFastDynamicBandwidthHarmonySearch(FastDynamicBandwidthHarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.45
        bandwidth = 0.1
        for _ in range(self.budget):
            candidate_solutions = []
            for _ in range(4):  # Process 4 candidate solutions concurrently
                candidate = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                candidate_solutions.append(candidate)
            fitness_values = Parallel(n_jobs=4)(delayed(func)(c) for c in candidate_solutions)
            best_index = np.argmin(fitness_values)
            if fitness_values[best_index] < func(self.harmony_memory[-1]):
                self.harmony_memory[-1] = candidate_solutions[best_index]
                self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
            for _ in range(2):
                for i in range(self.dim):
                    new_harmony_local = np.copy(self.harmony_memory[-1])
                    new_harmony_local[i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - bandwidth),
                                                             min(self.upper_bound, self.harmony_memory[-1][i] + bandwidth))
                    if func(new_harmony_local) < func(self.harmony_memory[-1]):
                        self.harmony_memory[-1] = new_harmony_local
            for i in range(self.dim):
                new_harmony_pitch = np.copy(self.harmony_memory[-1])
                new_harmony_pitch[i] = np.random.uniform(max(self.lower_bound, self.harmony_memory[-1][i] - bandwidth),
                                                         min(self.upper_bound, self.harmony_memory[-1][i] + bandwidth))
                if func(new_harmony_pitch) < func(self.harmony_memory[-1]):
                    self.harmony_memory[-1] = new_harmony_pitch
            bandwidth = min(0.5, max(0.01, bandwidth * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))))
        return self.harmony_memory[0]