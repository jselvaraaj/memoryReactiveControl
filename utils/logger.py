import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, name):
        self.writer = SummaryWriter(f"{name}")

    def plot_and_log_scalar(self, scalar_dict, tag: str):
        step_indices = sorted(scalar_dict.keys())

        mean_scalars = []
        std_scalars = []

        for step in step_indices:
            mean_scalar = np.mean(scalar_dict[step])
            std_scalar = np.std(scalar_dict[step])

            self.writer.add_scalar(f"Mean {tag}", mean_scalar, step)
            self.writer.add_scalar(f"STD {tag}", std_scalar, step)

            mean_scalars.append(mean_scalar)
            std_scalars.append(std_scalar)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(step_indices, mean_scalars, label=f"Mean {tag}", color="blue")
        plt.fill_between(step_indices,
                         np.array(mean_scalars) - np.array(std_scalars),
                         np.array(mean_scalars) + np.array(std_scalars),
                         color="blue", alpha=0.3, label="Standard Deviation")
        plt.xlabel("Step Index")
        plt.ylabel(tag)
        plt.title(f"Mean {tag} with Shaded Variance Across Episodes")
        plt.legend()
        plt.show()

    def __del__(self):
        self.writer.flush()
