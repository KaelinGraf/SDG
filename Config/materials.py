import os
import json
import re
import urllib.parse

# --- CONFIGURATION ---
EXISTING_JSON_PATH = "materials.json"
OUTPUT_JSON_PATH = "materials_updated.json"

# Keywords to capture relevant parameters
WEAR_KEYWORDS = [
    "scratch", "dirt", "dust", "rust", "oxid", "patina", 
    "damage", "wear", "pit", "bump", "roughness", 
    "brightness", "grime", "smudge", "abrasion", 
    "erosion", "dent", "crack", "brush", "groove",
    "variation", "tint", "color", "specular", "metallic",
    "reflection", "weight", "strength", "amount", "albedo",
    "anisotropy", "transmissive", "transmission"
]

def scan_mdl_for_parameters(file_path):
    found_params = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # --- 1. FIND FLOATS ---
        # Matches: "float name = 0.5" OR "name: 0.5" OR "name: 0.5f"
        float_def = re.findall(r'float\s+([a-zA-Z0-9_]+)\s*=\s*([\d\.]+)', content)
        float_assign = re.findall(r'\b([a-zA-Z0-9_]+)\s*:\s*([\d\.]+)(?:f)?', content)
        
        for param_name, default_val in float_def + float_assign:
            if any(k in param_name.lower() for k in WEAR_KEYWORDS):
                try:
                    val = float(default_val)
                    val = max(0.0, min(1.0, val)) # Clamp 0-1
                    
                    # --- SAFETY LIMITS (The Fix) ---
                    
                    # 1. ROUGHNESS: Never let it be 0.0 (Invisible mirror)
                    if "roughness" in param_name.lower():
                        min_val = max(0.15, val - 0.2) # Minimum 0.15
                        max_val = min(1.0, val + 0.3)
                    
                    # 2. AMOUNTS/DIRT: Allow full range 0-1
                    elif any(x in param_name.lower() for x in ["amount", "strength", "dirt", "smudge"]):
                         min_val = max(0.0, val - 0.3)
                         max_val = min(1.0, val + 0.3)
                    
                    # 3. GENERIC:
                    else:
                        min_val = max(0.0, val - 0.2)
                        max_val = min(1.0, val + 0.2)
                    
                    # Handle exact 0 or 1 defaults
                    if val == 0.0: max_val = 0.4
                    if val == 1.0: min_val = 0.6
                        
                    found_params[param_name] = [round(min_val, 3), round(max_val, 3)]
                except ValueError:
                    pass

        # --- 2. FIND COLORS ---
        # Matches: "color name = color(" OR "name: color("
        color_def = re.findall(r'color\s+([a-zA-Z0-9_]+)\s*=\s*color\(', content)
        color_assign = re.findall(r'\b([a-zA-Z0-9_]+)\s*:\s*color\(', content)
        
        for param_name in color_def + color_assign:
            if any(k in param_name.lower() for k in WEAR_KEYWORDS):
                found_params[param_name] = "color"

    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        
    return found_params

def get_local_path(mdl_url):
    if mdl_url.startswith("file://"):
        path = mdl_url[7:]
    else:
        path = mdl_url
    return urllib.parse.unquote(path)

def main():
    if not os.path.exists(EXISTING_JSON_PATH):
        print(f"Error: {EXISTING_JSON_PATH} not found.")
        return

    with open(EXISTING_JSON_PATH, 'r') as f:
        data = json.load(f)

    print(f"Scanning {len(data)} materials...")
    
    updated_count = 0

    for key, entry in data.items():
        mdl_url = entry.get("mdl_path", "")
        if not mdl_url: continue

        local_path = get_local_path(mdl_url)
        
        if os.path.exists(local_path):
            new_params = scan_mdl_for_parameters(local_path)
            
            if new_params:
                if "randomise" not in entry:
                    entry["randomise"] = {}
                
                entry["randomise"].update(new_params)
                updated_count += 1
        else:
            print(f"Warning: File not found for {key}: {local_path}")

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Success! Updated {updated_count} materials.")
    print(f"Saved to: {OUTPUT_JSON_PATH} (Rename this to materials.json)")

if __name__ == "__main__":
    main()