"""Entry point: run the monolithic-vs-modular update-cost simulation and plot it."""
from sim.plot import plot_results, print_summary
from sim.scenario import run_scenario

if __name__ == "__main__":
    total_time = 1000.0
    update_times = (200.0, 500.0, 800.0)

    metrics, update_times = run_scenario(total_time=total_time, update_times=update_times)

    print_summary(metrics, update_times)
    plot_results(metrics, update_times, total_time, out_path="results.png")
