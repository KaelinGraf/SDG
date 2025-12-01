import json
import os
import glob

# --- CONFIGURATION ---
# Change this to where you actually put the folder
LOCAL_ASSET_ROOT = "/home/kaelin/IsaacAssets" 
JSON_PATH = "Config/materials.json"
# ---------------------

def find_file(root_dir, target_filename):
    """
    Recursively searches for a file in the directory tree.
    Returns the absolute path if found, None otherwise.
    """
    # Fast walk to find the file
    for root, dirs, files in os.walk(root_dir):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

def main():
    if not os.path.exists(LOCAL_ASSET_ROOT):
        print(f"[ERROR] Asset root not found: {LOCAL_ASSET_ROOT}")
        print("Please check the path where you extracted the files.")
        return

    print(f"Scanning for assets in: {LOCAL_ASSET_ROOT} ...")

    try:
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {JSON_PATH}")
        return

    updated_count = 0
    
    for key, entry in data.items():
        # Extract just the filename (e.g. "Steel_Cast.mdl") from the long URL
        current_path = entry.get("mdl_path", "")
        filename = os.path.basename(current_path)
        
        # Search for this specific file in your local folder
        print(f"  Looking for: {filename}...", end=" ")
        local_path = find_file(LOCAL_ASSET_ROOT, filename)
        
        if local_path:
            # Convert to URI format for Isaac Sim (file://...)
            # We use absolute paths to be safe
            uri_path = f"file://{local_path}"
            
            entry["mdl_path"] = uri_path
            print(f"FOUND! \n    -> {local_path}")
            updated_count += 1
        else:
            print("NOT FOUND. (Keeping original URL)")

    # Save the updated file
    if updated_count > 0:
        with open(JSON_PATH, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\n[SUCCESS] Updated {updated_count} materials in {JSON_PATH} to use local storage.")
        print("You can now run scene_builder.py with zero network latency.")
    else:
        print("\n[WARNING] No matching local files were found. Check your folder structure.")

if __name__ == "__main__":
    main()