import json
import csv
import matplotlib.pyplot as plt
import pandas as pd

with open('build/benchmark_results.json', 'r') as f:
    data = json.load(f)

benchmarks = data.get("benchmarks", [])

results = []
for bench in benchmarks:
    name = bench.get("name", "unknown")
    iterations = bench.get("iterations", None)

    # Get the real time and convert from ns to microseconds
    real_time = bench.get("real_time", None)
    time_unit = bench.get("time_unit", "ns")
    if real_time is not None:
        if time_unit == "ns":
            duration_us = real_time / 1e3  # convert nanoseconds to microseconds
        elif time_unit == "us":
            duration_us = real_time
        else:
            duration_us = real_time  # fallback (you can adjust as needed)
    else:
        duration_us = None


    results.append({
        "Implementation": name,
        "Iterations": iterations,
        "Duration_us": duration_us,
    })

csv_filename = "data/benchmark_results.csv"
with open(csv_filename, "w", newline="") as csvfile:
    fieldnames = ["Implementation", "Iterations", "Duration_us"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"CSV data written to {csv_filename}")

df = pd.read_csv(csv_filename)

# Plot 1: Duration Plot (in microseconds)
plt.figure(figsize=(10, 6))
plt.bar(df['Implementation'], df['Duration_us'], color='skyblue')
plt.yscale('log')
plt.title('Duration for Each Optimization (μs)')
plt.xlabel('Optimization Variant')
plt.ylabel('Duration (μs) [log scale]')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/duration_plot.png')
plt.close()

# Plot 2: Iterations Plot
if df['Iterations'].notnull().all():
    plt.figure(figsize=(10, 6))
    plt.bar(df['Implementation'], df['Iterations'], color='lightgreen')
    plt.yscale('log')
    plt.title('Iterations for Each Optimization')
    plt.xlabel('Optimization Variant')
    plt.ylabel('Iterations [log scale]')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/iterations_plot.png')
    plt.close()
else:
    print("Iteration data missing for some variants; skipping iterations plot.")

print("Plots saved as: data/duration_plot.png, data/iterations_plot.png")
