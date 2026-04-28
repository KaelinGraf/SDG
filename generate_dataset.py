import json
import os
import subprocess
import random
from pathlib import Path

# Configuration
MASTER_OBJECTS_JSON = "./Config/objects.json"
BATCH_JSON_PATH = "./Config/current_batch.json"
PARTS_PER_BATCH = 10    # How many unique parts to load into RAM per run
FRAMES_PER_BATCH = 200  # How many frames to generate before restarting Isaac Sim

def main():
    # 1. Load your massive list of thousands of objects
    with open(MASTER_OBJECTS_JSON, 'r') as f:
        master_config = json.load(f)
    
    all_part_keys = list(master_config["parts"].keys())
    print(f"Loaded {len(all_part_keys)} total parts from master config.")

    # 2. The Infinite Generation Loop
    batch_index = 0
    while True:
        print(f"\n{'='*50}\nStarting Batch {batch_index}\n{'='*50}")
        
        # 3. Pick a random subset of parts for this batch
        selected_keys = random.sample(all_part_keys, PARTS_PER_BATCH)
        
        # # 4. Create a mini-config containing ONLY these parts
        batch_config = {}
        batch_config['bins'] = master_config.get('bins', {})  # Include bins if needed
        batch_config["parts"] = {key: master_config["parts"][key] for key in selected_keys}
        
        
        # Write it to disk so the worker can read it
        with open(BATCH_JSON_PATH, 'w') as f:
            json.dump(batch_config, f, indent=4)
            
        # 5. Launch Isaac Sim as a Subprocess
        # This completely isolates the memory. When it finishes, VRAM goes back to 0.
        command = [
            "python", "scene_builder.py", 
            "--scene_name", "default_scene", 
            "--iters", str(FRAMES_PER_BATCH),
            "--batch_file", BATCH_JSON_PATH, # Change to BATCH_JSON_PATH if you want to use the subset config
        ]
        
        try:
            # This blocks until Isaac Sim finishes generating the 500 frames and closes
            subprocess.run(command, check=True)
            print(f"Batch {batch_index} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Isaac Sim crashed during Batch {batch_index}. Orchestrator recovering...")
            # It crashed? Who cares! The orchestrator survives and just starts the next batch.
            
        batch_index += 1

if __name__ == "__main__":
    main()