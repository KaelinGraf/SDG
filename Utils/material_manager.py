#methods for templating, randomising and applying realistic materials to USD prims
from isaacsim.sensors.rtx import apply_nonvisual_material
from isaacsim.core.api.materials import OmniPBR
from pxr import UsdShade, Sdr, Sdf  
import omni.kit.commands
import json     
import random
import omni
import carb
import os
import omni.kit.material.library

class MaterialManager:
    def __init__(self):
        self.asset_root = "/home/kaelin/Documents/mdl"
        self._register_mdl_search_path(self.asset_root)
        self.templates=json.load(open("Config/materials.json"))
        if self.templates is None:
            raise ValueError("Failed to load materials.json")
        
        self.stage = omni.usd.get_context().get_stage()
        self.materials = list(self.templates.keys()) #iterable of available materials for randomisation
        self.materials_in_scene = [] #stores the paths of materials that have been created in the scene 
        self.param_map = {
            "roughness": "inputs:reflection_roughness_constant",
            "metallic": "inputs:metallic_constant",
            "specular": "inputs:specular_level", # Some MDLs use this
            "weight": "inputs:reflection_weight"  # Fallback for some materials
        }
    def reset(self):
        """Call this at the start of every data_generator_loop iteration"""
        self.active_scene_materials = []
    def _register_mdl_search_path(self, local_root_path):

        """Fixes dependency resolution by adding local folder to renderer paths"""
        if not os.path.exists(local_root_path): return
        settings = carb.settings.get_settings()
        key = "/renderer/mdl/searchPaths"
        current = settings.get(key)
        
        # Add path if not present (handles list or string formats)
        if isinstance(current, list):
            if local_root_path not in current:
                current.append(local_root_path)
                settings.set(key, current)
        else:
            if not current or local_root_path not in current:
                new_paths = f"{current}:{local_root_path}" if current else local_root_path
                settings.set_string(key, new_paths)
            
        print(f"[MaterialManager] Added MDL Search Path: {local_root_path}")

    def _sanitize_path(self, raw_url):
        """
        Converts absolute paths to Relative Paths based on the Asset Root.
        This forces the MDL compiler to use package-based resolution (fixing C120 errors).
        """
        # 1. Strip URI prefix
        if raw_url.startswith("file://"):
            clean_path = raw_url.replace("file://", "")
        else:
            clean_path = raw_url

        # 2. Relativize
        # If the path starts with our root (e.g. /home/kaelin/IsaacAssets/Materials/...)
        # We cut it down to (Materials/...)
        if clean_path.startswith(self.asset_root):
            # +1 removes the leading slash
            rel_path = clean_path[len(self.asset_root):].lstrip("/")
            return rel_path
            
        return clean_path
    def get_template(self,template_name):
        result = self.templates.get(template_name,None)
        if result is None:
            carb.log_warn(f"Template {template_name} not found")
            raise ValueError(f"Template {template_name} not found")
        return result
    def populate_materials(self,n=1):
        """creates n materials, stores their paths in a list"""
        for i in range(n):
            self.materials_in_scene.append(self.create_material())
    def create_material(self,template=None):

        """create a single material primitive and return the path"""

        if template is None:
            template_key = random.choice(self.materials)
        else:
            template_key = template
            
        template_data = self.get_template(template_key)
        
        # Unique name for this specific application
        material_idx = len(self.materials_in_scene)
        mat_name = f"{template_data['name']}"
        mat_path = f"/World/Looks/{mat_name}"
        mdl_url = self._sanitize_path(template_data['mdl_path'])

        # 1. Create the Material
        mat_prim_path = omni.kit.material.library.create_mdl_material(
            stage=self.stage,
            mtl_url=mdl_url,
            mtl_name=mat_name,
            on_create_fn=self.on_created_mdl
        )
        if mat_prim_path is not None:
            carb.log_info(f"[MaterialManager] Created material {mat_name} at {mat_prim_path}")
            self.materials_in_scene.append(mat_prim_path)

        # 2. Smart Wait for Compilation
        # We check if the shader is in the registry. 
        # We search by filename because 'file:///...' might differ from registry's internal path.
        target_filename = os.path.basename(mdl_url)
        registry = Sdr.Registry()
        
        # Fast check: is it already there?
        found = False
        if self._is_shader_loaded(registry, target_filename):
            found = True
        else:
            # Wait loop
            # print(f"Compiling {target_filename}...")
            for _ in range(50): # Wait up to ~1 sec (50 frames is plenty for local files)
                omni.kit.app.get_app().update()
                if self._is_shader_loaded(registry, target_filename):
                    found = True
                    break
            
        if not found:
            # This prevents the crash. If it times out, we skip randomization but don't error out.
            carb.log_warn(f"Shader {target_filename} timed out. Skipping randomization.")
        
        # 3. Randomize Inputs (Only if shader was found)
        if found and "randomise" in template_data:
            mat_prim = self.stage.GetPrimAtPath(mat_prim_path)
            shader = UsdShade.Shader(mat_prim)
            
            for param_key, bounds in template_data["randomise"].items():
                usd_input = self.param_map.get(param_key)
                if usd_input:
                    val = random.uniform(bounds[0], bounds[1])
                    # Sdf is now imported, so this will work
                    shader.CreateInput(usd_input, Sdf.ValueTypeNames.Float).Set(val)

        # 4. Apply Non-Visual Attributes
        if "non_visual" in template_data and len(template_data['non_visual']) == 3:
            # Re-get prim to be safe
            mat_prim = self.stage.GetPrimAtPath(mat_prim_path)
            apply_nonvisual_material(
                mat_prim,
                template_data['non_visual'][0],
                template_data['non_visual'][1],
                template_data['non_visual'][2]
            )
        return mat_prim_path

    def bind_material(self, mat_prim_path=None, prim_path=None):
        """
        Binds a material to a prim.
        """
        if prim_path is None:
            raise ValueError("prim_path must be specified")
        
        prim = self.stage.GetPrimAtPath(prim_path)

        if mat_prim_path is None:
            mat_prim_path = random.choice(self.materials_in_scene)
        
        omni.kit.material.library.bind_material_to_selected_prims(
            material_prim_path=mat_prim_path,
            paths=[prim_path]
        )
    def create_and_bind(self, template=None, prim_path=None):
        """
        Creates a new material instance, randomizes it, and binds it to the prim.
        """
        if prim_path is None:
            raise ValueError("prim_path must be specified")
        
        prim = self.stage.GetPrimAtPath(prim_path)
        
        if template is None:
            template_key = random.choice(self.materials)
        else:
            template_key = template
            
        template_data = self.get_template(template_key)
        
        # Unique name for this specific application
        material_idx = len(self.materials_in_scene)
        mat_name = f"{template_data['name']}"
        mat_path = f"/World/Looks/{mat_name}"
        mdl_url = template_data['mdl_path']

        # 1. Create the Material
        mat_prim_path = omni.kit.material.library.create_mdl_material(
            stage=self.stage,
            mtl_url=mdl_url,
            mtl_name=mat_name,
            on_create_fn=self.on_created_mdl
        )
        if mat_prim_path is not None:
            carb.log_info(f"[MaterialManager] Created material {mat_name} at {mat_prim_path}")
            self.materials_in_scene.append(mat_prim_path)

        # 2. Smart Wait for Compilation
        # We check if the shader is in the registry. 
        # We search by filename because 'file:///...' might differ from registry's internal path.
        target_filename = os.path.basename(mdl_url)
        registry = Sdr.Registry()
        
        # Fast check: is it already there?
        found = False
        if self._is_shader_loaded(registry, target_filename):
            found = True
        else:
            # Wait loop
            # print(f"Compiling {target_filename}...")
            for _ in range(50): # Wait up to ~1 sec (50 frames is plenty for local files)
                omni.kit.app.get_app().update()
                if self._is_shader_loaded(registry, target_filename):
                    found = True
                    break
            
        if not found:
            # This prevents the crash. If it times out, we skip randomization but don't error out.
            carb.log_warn(f"Shader {target_filename} timed out. Skipping randomization.")
        
        # 3. Randomize Inputs (Only if shader was found)
        if found and "randomise" in template_data:
            mat_prim = self.stage.GetPrimAtPath(mat_prim_path)
            shader = UsdShade.Shader(mat_prim)
            
            for param_key, bounds in template_data["randomise"].items():
                usd_input = self.param_map.get(param_key)
                if usd_input:
                    val = random.uniform(bounds[0], bounds[1])
                    # Sdf is now imported, so this will work
                    shader.CreateInput(usd_input, Sdf.ValueTypeNames.Float).Set(val)

        # 4. Apply Non-Visual Attributes
        if "non_visual" in template_data and len(template_data['non_visual']) == 3:
            # Re-get prim to be safe
            mat_prim = self.stage.GetPrimAtPath(mat_prim_path)
            apply_nonvisual_material(
                mat_prim,
                template_data['non_visual'][0],
                template_data['non_visual'][1],
                template_data['non_visual'][2]
            )

        # 5. Bind
        #mat_prim = self.stage.GetPrimAtPath(mat_path)
        
        omni.kit.material.library.bind_material_to_selected_prims(
            material_prim_path=mat_prim_path,
            paths=[prim_path]
        )
    def on_created_mdl(self,mat_prim):
        carb.log_info("[MaterialManager] Created MDL material")
    def _is_shader_loaded(self, registry, filename):
        """Helper to find if a shader is loaded by checking identifiers for the filename."""
        # This is expensive to check all, but safe. 
        # For better perf, you can try registry.GetNodeByIdentifier(full_path) first.
        
        # 1. Try exact match (Fastest)
        # Sdr often strips 'file://' so we try raw path if URL provided
        if registry.GetNodeByIdentifier(filename): 
            return True
            
        # 2. Fuzzy search (Slower but robust for file URI issues)
        # Only do this if you are getting timeouts
        for uri in registry.GetNodeIdentifiers():
            if filename in uri:
                return True
        return False