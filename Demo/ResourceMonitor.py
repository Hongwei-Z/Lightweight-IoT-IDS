import time
import psutil
from codecarbon import EmissionsTracker
import logging
from typing import Dict


class ResourceMonitor:
    # Class for monitoring and reporting system resource usage.

    def __init__(self):
        # Initialize the resource monitor.
        logging.basicConfig(level=logging.ERROR)
        self.tracker = None
        self.start_time = None

    def start(self) -> None:
        # Start monitoring resources.

        self.tracker = EmissionsTracker(save_to_file=False, allow_multiple_runs=True)
        self.tracker.start()
        self.start_time = time.time()

        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        print(f"\nStart - CPU Usage: {cpu_usage}%")
        print(f"Start - Memory Usage: {memory_usage}%")

    def stop(self) -> Dict[str, float]:
        # Stop monitoring and return resource usage statistics.

        if not self.tracker or not self.start_time:
            raise RuntimeError("Resource monitoring was not started")

        self.tracker.stop()
        end_time = time.time()

        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        energy_consumed_kwh = self.tracker.final_emissions_data.energy_consumed
        co2_emission_kg = self.tracker.final_emissions_data.emissions
        execution_time = end_time - self.start_time

        # Print resource usage information
        line = "*" * 29
        print('\n' + line)
        print("Resource Usage:")
        print(f"Execution Time: {execution_time:.6f}s")
        print(f"Energy Consumed: {energy_consumed_kwh:.6f}kWh")
        print(f"COâ‚‚ Emission: {co2_emission_kg:.6f}kg")
        print(f"End - CPU Usage: {cpu_usage}%")
        print(f"End - Memory Usage: {memory_usage}%")
        print(line + '\n')

        # Return metrics as a dictionary
        return {
            "execution_time": execution_time,
            "energy_consumed_kwh": energy_consumed_kwh,
            "co2_emission_kg": co2_emission_kg,
            "final_cpu_usage": cpu_usage,
            "final_memory_usage": memory_usage
        }


def monitor_resources():
    # Legacy function for starting resource monitoring.
    monitor = ResourceMonitor()
    monitor.start()
    return monitor.tracker, monitor.start_time


def stop_monitoring(tracker, start_time):
    # Legacy function for stopping resource monitoring.
    temp_monitor = ResourceMonitor()
    temp_monitor.tracker = tracker
    temp_monitor.start_time = start_time
    return temp_monitor.stop()

