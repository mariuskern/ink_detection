import numpy as np


class PredictionMerger:
    def __init__(self):
        self.buffers = {}

    def add(self, segment, x, y, prediction, shape):
        if segment not in self.buffers:
            H, W = shape[1], shape[2]

            self.buffers[segment] = {
                "prob_sum": np.zeros((H, W), dtype=np.float32),
                "prob_sq_sum": np.zeros((H, W), dtype=np.float32),
                "weight_sum": np.zeros((H, W), dtype=np.float32),
            }

        buffer = self.buffers[segment]

        h, w = prediction.shape
        H, W = buffer["prob_sum"].shape

        h_valid = min(h, H - y)
        w_valid = min(w, W - x)

        if h_valid <= 0 or w_valid <= 0:
            return

        prediction = prediction[:h_valid, :w_valid]
        weight = self._gaussian_weight(h)[:h_valid, :w_valid]

        buffer["prob_sum"][y:y+h_valid, x:x+w_valid] += prediction * weight
        buffer["prob_sq_sum"][y:y+h_valid, x:x+w_valid] += (prediction ** 2) * weight
        buffer["weight_sum"][y:y+h_valid, x:x+w_valid] += weight

    def get_predictions(self):
        results = {}

        for segment, buffer in self.buffers.items():
            prediction = buffer["prob_sum"] / np.maximum(buffer["weight_sum"], 1e-6)

            second_moment = buffer["prob_sq_sum"] / np.maximum(buffer["weight_sum"], 1e-6)
            variance = np.maximum(second_moment - prediction**2, 0)
            uncertainty = np.sqrt(variance)

            confidence = np.maximum(prediction, 1 - prediction)

            results[segment] = {
                "prediction": prediction,
                "uncertainty": uncertainty,
                "confidence": confidence,
            }

        return results

    def clear(self):
        self.buffers.clear()


    @staticmethod
    def _gaussian_weight(size, sigma_scale=1/6):
        ax = np.arange(size) - (size - 1) / 2
        xx, yy = np.meshgrid(ax, ax)

        sigma = size * sigma_scale
        g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2)
        )

        return (g / g.max()).astype(np.float32)