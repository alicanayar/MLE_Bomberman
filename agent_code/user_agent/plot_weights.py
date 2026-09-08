
import re
import ast
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Configurations ---
log_file_path = "C:/Users/msı/Desktop/Universitat Heidelberg/MLE/bomberman/bomberman_rl/agent_code/user_agent/logs/user_agent.log"  # Replace with your actual log file path

feature_names = [
    "Safety Dist",
    "Steps vs Bomb",
    "Corridor Depth",
    "Blast UP",
    "Blast DOWN",
    "Blast LEFT",
    "Blast RIGHT",
    "Valid Action"  # Rename according to your actual 8th feature
]

# --- 2. Extract Weights from Log File ---
weights_history = []
pattern = re.compile(r"Updated Weights (\[.*?\])")

try:
    with open(log_file_path, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # Convert string representation of list "[w1, w2, ...]" to Python list
                weight_list = ast.literal_eval(match.group(1))
                if len(weight_list) == 8:
                    weights_history.append(weight_list)

except FileNotFoundError:
    print(f"Error: Log file '{log_file_path}' not found. Please check the file path.")
    exit()

if not weights_history:
    print("No matching 'Updated Weights' entries found in the log file.")
    exit()

# Convert list to NumPy array
weights_array = np.array(weights_history)

print(f"Successfully extracted {len(weights_array)} weight updates.")
print(f"Array Shape: {weights_array.shape}")

# --- 3. Plotting Logic ---

if weights_array.ndim == 1 or weights_array.shape[0] == 1:
    # CASE A: Only 1 log entry found -> Plot Bar Chart
    weights_1d = weights_array.flatten()
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        feature_names, 
        weights_1d, 
        color=['teal' if w >= 0 else 'crimson' for w in weights_1d]
    )
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    
    for bar in bars:
        yval = bar.get_height()
        offset = 0.02 if yval >= 0 else -0.05
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, 
            yval + offset, 
            f'{yval:.2f}', 
            ha='center', 
            va='bottom' if yval >= 0 else 'top', 
            fontsize=9
        )

    plt.title("Logged Feature Weights Snapshot", fontsize=14, fontweight='bold')
    plt.ylabel("Weight Value", fontsize=11)
    plt.xticks(rotation=25, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

else:
    # CASE B: Multiple log entries found -> Plot 8 Subplots Over Time
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for i in range(8):
        axes[i].plot(weights_array[:, i], color='tab:blue', linewidth=1.5)
        axes[i].axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
        axes[i].set_title(feature_names[i], fontsize=11, fontweight='bold')
        axes[i].grid(True, linestyle='--', alpha=0.5)

    fig.text(0.5, 0.01, 'Log Updates / Episodes', ha='center', fontsize=12)
    fig.text(0.01, 0.5, 'Weight Value', va='center', rotation='vertical', fontsize=12)

    plt.suptitle("Feature Weights Convergence Extracted from Log File", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])
    plt.show()

# --- 4. Print Latest Weights to Console ---
print("\n--- LATEST FEATURE WEIGHTS ---")
latest_weights = weights_array[-1] if weights_array.ndim > 1 else weights_array.flatten()
for name, val in zip(feature_names, latest_weights):
    print(f"{name:<16}: {val:.4f}")