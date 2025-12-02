import os
import json
import re

# --- CONFIGURATION ---
EXISTING_JSON_PATH = "materials.json"
OUTPUT_JSON_PATH = "materials_updated.json"
SEARCH_DIR = "/home/kaelin/Documents/mdl/vMaterials_2/Metal"
URL_PREFIX = "file:///home/kaelin/Documents/mdl/vMaterials_2/Metal/"

# Keywords to look for in MDL parameters
WEAR_KEYWORDS = [
    "scratch", "dirt", "dust", "rust", "oxid", "patina", 
    "damage", "wear", "pit", "bump", "roughness_variation",
    "brightness_variation", "grime", "smudge"
]

def get_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_tags_from_name(name):
    """(Same categorization logic as before)"""
    n = name.lower()
    if "patina" in n or "antique" in n:
        if "copper" in n or "bronze" in n: return ["oxidized_Bronze_Patina", "none", "emissive"]
    if "rust" in n or "oxidized" in n or "pitted" in n:
        if "iron" in n: return ["oxidized_iron", "none", "emissive"]
        if "steel" in n: return ["oxidized_steel", "none", "emissive"]
    if "aluminum" in n or "aluminium" in n: return ["aluminum", "none", "emissive"]
    if "brass" in n: return ["brass", "none", "emissive"]
    if "bronze" in n or "copper" in n: return ["bronze", "none", "emissive"]
    if "iron" in n: return ["iron", "none", "emissive"]
    if any(x in n for x in ["silver", "chrome", "chromium", "nickel", "platinum", "mercury"]): return ["silver", "none", "emissive"]
    if "tin" in n: return ["tin", "none", "emissive"]
    return ["steel", "none", "emissive"]

def scan_mdl_for_parameters(file_path):
    """
    Parses the MDL file text to find 'float parameter_name' that matches our keywords.
    Returns a dict of found parameters with default randomization ranges.
    """
    found_params = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Regex to find float parameters: "float param_name = 0.5"
        # We capture the name (group 1) and the default value (group 2)
        pattern = re.compile(r'float\s+([a-zA-Z0-9_]+)\s*=\s*([\d\.]+)')
        matches = pattern.findall(content)
        
        for param_name, default_val in matches:
            # Check if this parameter is interesting (contains a keyword)
            if any(k in param_name.lower() for k in WEAR_KEYWORDS):
                try:
                    val = float(default_val)
                    # Create a sensible range around the default, clamped 0-1
                    min_val = max(0.0, val - 0.2)
                    max_val = min(1.0, val + 0.4)
                    
                    # Special overrides for common "amount" params
                    if "amount" in param_name or "intensity" in param_name:
                        min_val, max_val = 0.0, 0.8
                        
                    found_params[param_name] = [round(min_val, 3), round(max_val, 3)]
                except ValueError:
                    pass
                    
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        
    return found_params

def main():
    if os.path.exists(EXISTING_JSON_PATH):
        with open(EXISTING_JSON_PATH, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    print(f"Loaded {len(data)} existing materials.")
    
    # Existing paths check
    existing_paths = set(entry.get("mdl_path", "") for entry in data.values())

    files = [f for f in os.listdir(SEARCH_DIR) if f.endswith(".mdl")]
    files.sort()
    
    added_count = 0
    updated_count = 0

    for filename in files:
        full_url = f"{URL_PREFIX}{filename}"
        full_path = os.path.join(SEARCH_DIR, filename)
        
        # 1. GENERATE BASE ENTRY (if new)
        if full_url not in existing_paths:
            base_name = os.path.splitext(filename)[0]
            json_key = get_snake_case(base_name)
            if json_key in data: json_key = f"{json_key}_v2"
            
            new_entry = {
                "mdl_path": full_url,
                "name": base_name,
                "non_visual": get_tags_from_name(base_name),
                "randomise": {
                    "specular": [0.2, 1.0],
                    "roughness": [0.1, 0.7],
                    "metallic": [0.8, 1.0] 
                }
            }
            data[json_key] = new_entry
            added_count += 1
        else:
            # Find the existing key for this URL to update it
            json_key = next((k for k, v in data.items() if v["mdl_path"] == full_url), None)

        # 2. SCAN AND INJECT WEAR PARAMETERS
        if json_key:
            extra_params = scan_mdl_for_parameters(full_path)
            if extra_params:
                # Merge into existing randomize block
                data[json_key]["randomise"].update(extra_params)
                updated_count += 1
                # print(f"  -> Added params to {json_key}: {list(extra_params.keys())}")

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"\nSuccess! Added {added_count} new materials.")
    print(f"Updated {updated_count} materials with wear parameters.")
    print(f"Saved to: {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()