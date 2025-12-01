# Save as setup_mdl_library.py and run it
from isaacsim import SimulationApp
import warp
simulation_app = SimulationApp({
    "headless": True,
}) 
import os
import shutil
import carb

# --- CONFIGURATION ---
TARGET_DIR = "/home/kaelin/Documents/mdl"
SOURCE_ASSETS = "/home/kaelin/IsaacAssets"  # Where you extracted your download
# ---------------------

def create_symlink(source, destination):
    if os.path.exists(destination):
        print(f"[SKIP] {destination} already exists.")
        return
    
    # Check if source exists
    if not os.path.exists(source):
        print(f"[ERROR] Source does not exist: {source}")
        return

    try:
        os.symlink(source, destination)
        print(f"[SUCCESS] Linked: {source} -> {destination}")
    except OSError as e:
        print(f"[FAIL] Could not link {source}: {e}")

def main():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"Created directory: {TARGET_DIR}")

    # 1. FIND INTERNAL NVIDIA CORE DEFINITIONS
    # This finds the missing 'definitions.mdl' hidden deep in your pip install
    isaac_root = carb.tokens.get_tokens_interface().resolve("${isaac_sim}")
    internal_nvidia_path = None
    
    print("Searching for internal Core Definitions...")
    for root, dirs, files in os.walk(isaac_root):
        # We are looking for .../mdl/core/import/nvidia
        if "definitions.mdl" in files and root.endswith("nvidia/core"):
            # We want the folder containing 'nvidia'
            # root is .../nvidia/core. 
            # parent is .../nvidia
            # grandparent is .../import (THIS is what we want to link 'nvidia' from? No.)
            # We want to link the 'nvidia' folder itself into Documents/mdl/nvidia
            
            # root = /.../mdl/core/import/nvidia/core
            # We want /.../mdl/core/import/nvidia
            
            internal_nvidia_path = os.path.dirname(root) # This gets '.../nvidia'
            print(f"FOUND Internal Core: {internal_nvidia_path}")
            break
            
    # 2. LINK NVIDIA CORE
    if internal_nvidia_path:
        target_nvidia = os.path.join(TARGET_DIR, "NVIDIA")
        create_symlink(internal_nvidia_path, target_nvidia)
    else:
        print("[CRITICAL] Could not find internal NVIDIA definitions!")

    # 3. LINK YOUR vMATERIALS
    # You have /home/kaelin/IsaacAssets/Materials/vMaterials_2
    # We want /home/kaelin/Documents/mdl/Materials/vMaterials_2
    # OR simpler: /home/kaelin/Documents/mdl/vMaterials_2 (if you adjust imports)
    # BUT standard is typically: Documents/mdl/Materials/...
    
    # Let's mirror your IsaacAssets structure exactly
    # Link "Materials" from IsaacAssets to Documents/mdl/Materials
    source_materials = os.path.join(SOURCE_ASSETS, "Materials") # Adjust if your path differs
    if not os.path.exists(source_materials):
         # Try finding where vMaterials_2 is
         if os.path.exists(os.path.join(SOURCE_ASSETS, "vMaterials_2")):
             # If you extracted it flat
             source_materials = SOURCE_ASSETS
         elif os.path.exists(os.path.join(SOURCE_ASSETS, "Isaac", "Materials")):
             source_materials = os.path.join(SOURCE_ASSETS, "Isaac", "Materials")

    target_materials = os.path.join(TARGET_DIR, "Materials")
    
    # We might need to make the parent folder structure if it doesn't align perfectly
    # Safe bet: Link the vMaterials_2 folder specifically
    
    # Find vMaterials_2 in source
    vmat_source = None
    for root, dirs, files in os.walk(SOURCE_ASSETS):
        if "vMaterials_2" in dirs:
            vmat_source = os.path.join(root, "vMaterials_2")
            break
            
    if vmat_source:
        # Create Materials folder in target
        if not os.path.exists(target_materials):
            os.makedirs(target_materials)
            
        target_vmat = os.path.join(target_materials, "vMaterials_2")
        create_symlink(vmat_source, target_vmat)
        
        # ALSO Link Base if it exists
        base_source = os.path.join(os.path.dirname(vmat_source), "Base")
        if os.path.exists(base_source):
             target_base = os.path.join(target_materials, "Base")
             create_symlink(base_source, target_base)
             
    else:
        print("[ERROR] Could not find vMaterials_2 in your source assets.")

    print("\n--- DONE ---")
    print(f"Your MDL library is ready at: {TARGET_DIR}")
    print("Structure should contain:")
    print(f"  {TARGET_DIR}/NVIDIA  (Linked from Internal)")
    print(f"  {TARGET_DIR}/Materials/vMaterials_2 (Linked from Download)")

if __name__ == "__main__":
    main()