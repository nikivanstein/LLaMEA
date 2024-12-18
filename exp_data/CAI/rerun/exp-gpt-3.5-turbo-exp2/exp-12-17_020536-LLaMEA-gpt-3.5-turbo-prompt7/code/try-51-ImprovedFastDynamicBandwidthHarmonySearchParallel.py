import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ImprovedFastDynamicBandwidthHarmonySearchParallel(FastDynamicBandwidthHarmonySearch):
    def __call__(self, func):
        pitch_rate = 0.45
        bandwidth = 0.1
        with ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(self.budget):
                futures.append(executor.submit(self._optimize_harmony, func, pitch_rate, bandwidth))
                pitch_rate = max(0.01, pitch_rate * 0.97 * 1.3 * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))
            for future in futures:
                result = future.result()
                if func(result) < func(self.harmony_memory[-1]):
                    self.harmony_memory[-1] = result
                    self.harmony_memory = self.harmony_memory[np.argsort([func(h) for h in self.harmony_memory])]
                bandwidth = min(0.5, max(0.01, bandwidth * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1]))))
        return self.harmony_memory[0]
    
    def _optimize_harmony(self, func, pitch_rate, bandwidth):
        new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        for _ in range(2):
            pitch_rate = max(0.01, pitch_rate * 0.97 * 1.3 * (1 + 0.1 * np.mean(np.abs(self.harmony_memory[0] - self.harmony_memory[-1])))
            for i in range(self.dim):
                if np.random.rand() < pitch_rate:
                    new_harmony[i] = np.random.choice(self.harmony_memory[:, i])
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
        return new_harmony