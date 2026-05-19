import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


RESULTS_DIR = Path("../results")
FIGURES_DIR = Path("../figures")


class BenchmarkVisualizer:

    def __init__(self, results_dir=RESULTS_DIR):

        self.results_dir = Path(results_dir)


    def list_available_csvs(self):
        """
        List all benchmark CSV files.
        """

        csv_files = list(
            self.results_dir.glob("*.csv")
        )

        if not csv_files:

            print("No CSV files found.")
            return []

        print("\nAvailable benchmark files:\n")

        for idx, file in enumerate(csv_files):

            print(f"[{idx}] {file.name}")

        return csv_files


    def load_selected_csvs(self):
        """
        Interactive CSV selection.
        """

        csv_files = self.list_available_csvs()

        if not csv_files:
            return None

        selection = input(
            "\nEnter CSV indexes separated by commas: "
        )

        indexes = [
            int(x.strip())
            for x in selection.split(",")
        ]

        selected_files = [
            csv_files[i]
            for i in indexes
        ]

        dataframes = []

        for file in selected_files:

            df = pd.read_csv(file)

            dataframes.append(df)

        combined_df = pd.concat(
            dataframes,
            ignore_index=True
        )

        return combined_df


    def show_summary_table(self, df):
        """
        Show benchmark summary.
        """

        summary = df[[
            "model",
            "precision",
            "ram_usage_gb",
            "tokens_per_second",
            "inference_time"
        ]]

        print("\nBenchmark Summary:\n")

        print(summary)


    def plot_ram_usage(self, df, save=False):
        """
        Plot RAM usage grouped by model.
        """

        pivot_df = df.pivot(
            index="precision",
            columns="model",
            values="ram_usage_gb"
        )

        ax = pivot_df.plot(
            kind="bar",
            figsize=(10, 6)
        )

        ax.set_xlabel("Precision")
        ax.set_ylabel("RAM Usage (GB)")
        ax.set_title(
            "RAM Usage by Model and Precision"
        )

        plt.xticks(rotation=0)

        plt.legend(title="Model")

        plt.tight_layout()

        if save:

            FIGURES_DIR.mkdir(
                exist_ok=True
            )

            plt.savefig(
                FIGURES_DIR / "ram_usage_models.png",
                bbox_inches="tight"
            )

        plt.show()


    def plot_tokens_per_second(
        self,
        df,
        save=False
    ):
        """
        Plot throughput grouped by model.
        """

        pivot_df = df.pivot(
            index="precision",
            columns="model",
            values="tokens_per_second"
        )

        ax = pivot_df.plot(
            kind="bar",
            figsize=(10, 6)
        )

        ax.set_xlabel("Precision")
        ax.set_ylabel("Tokens per Second")

        ax.set_title(
            "Throughput by Model and Precision"
        )

        plt.xticks(rotation=0)

        plt.legend(title="Model")

        plt.tight_layout()

        if save:

            FIGURES_DIR.mkdir(
                exist_ok=True
            )

            plt.savefig(
                FIGURES_DIR / "throughput_models.png",
                bbox_inches="tight"
            )

        plt.show()


    def plot_inference_time(
        self,
        df,
        save=False
    ):
        """
        Plot inference latency grouped by model.
        """

        pivot_df = df.pivot(
            index="precision",
            columns="model",
            values="inference_time"
        )

        ax = pivot_df.plot(
            kind="bar",
            figsize=(10, 6)
        )

        ax.set_xlabel("Precision")

        ax.set_ylabel(
            "Inference Time (sec)"
        )

        ax.set_title(
            "Latency by Model and Precision"
        )

        plt.xticks(rotation=0)

        plt.legend(title="Model")

        plt.tight_layout()

        if save:

            FIGURES_DIR.mkdir(
                exist_ok=True
            )

            plt.savefig(
                FIGURES_DIR / "latency_models.png",
                bbox_inches="tight"
            )

        plt.show()


if __name__ == "__main__":

    visualizer = BenchmarkVisualizer()

    df = visualizer.load_selected_csvs()

    if df is not None:

        visualizer.show_summary_table(df)

        visualizer.plot_ram_usage(
            df,
            save=True
        )

        visualizer.plot_tokens_per_second(
            df,
            save=True
        )

        visualizer.plot_inference_time(
            df,
            save=True
        )

