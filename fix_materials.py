import json
import os

# --- Configuration ---
INPUT_FILE = "Config/materials.json"  # Adjust if your file is elsewhere
OUTPUT_FILE = "Config/materials_fixed.json"

# The new local root for your MDLs
NEW_ROOT = "/home/kaelin/Documents/mdl"

# The part of the old path we want to replace
# We look for where the folder structure diverges. 
# Old: .../IsaacAssets/Materials/vMaterials_2/...
# New: .../Documents/mdl/vMaterials_2/...
OLD_PREFIX_MARKER = "/Materials/" 

def fix_materials():
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"Processing {len(data)} materials...")
    
    modified_count = 0

    for key, material in data.items():
        old_path = material.get("mdl_path", "")
        
        # 1. Update the Path
        # We strip the "file://" prefix first to handle path logic cleanly
        clean_path = old_path.replace("file://", "")
        
        if OLD_PREFIX_MARKER in clean_path:
            # Split path at "/Materials/"
            # internal_path will be "vMaterials_2/Metal/..." or "Base/Metals/..."
            _, internal_path = clean_path.split(OLD_PREFIX_MARKER, 1)
            
            # Construct new full path
            # Result: /home/kaelin/Documents/mdl/vMaterials_2/Metal/...
            new_path_raw = os.path.join(NEW_ROOT, internal_path)
            
            # Add file:// prefix back for consistency
            new_mdl_path = f"file://{new_path_raw}"
            
            material["mdl_path"] = new_mdl_path
        else:
            print(f"Warning: '{key}' path did not contain '{OLD_PREFIX_MARKER}'. Skipping path update.")
            new_mdl_path = old_path

        # 2. Update the Name (Fixing Casing)
        # Extract filename: "Aluminum_Brushed.mdl"
        filename = os.path.basename(new_mdl_path)
        # Remove extension: "Aluminum_Brushed"
        proper_name, _ = os.path.splitext(filename)
        
        # Update the 'name' field to match the PascalCase filename
        if material["name"] != proper_name:
            print(f"  [{key}] Renaming '{material['name']}' -> '{proper_name}'")
            material["name"] = proper_name
            modified_count += 1
            
    # Save the result
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"\nSuccess! Saved fixed config to '{OUTPUT_FILE}'.")
    print(f"Updated paths and names for {modified_count} entries.")

if __name__ == "__main__":
    fix_materials()