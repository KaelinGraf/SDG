#methods for templating, randomising and applying realistic materials to USD prims
from isaacsim.sensors.rtx import apply_nonvisual_material
from isaacsim.core.api.materials import OmniPBR
from pxr import UsdShade, Sdr, Sdf, UsdGeom, Gf
import omni.kit.commands
import json     
import random
import omni
import carb
import os
import omni.kit.material.library
import colorsys
from omni.isaac.core.utils import prims


class MaterialManager:
    def __init__(self):
        
        self.templates=json.load(open("Config/materials.json"))
        if self.templates is None:
            raise ValueError("Failed to load materials.json")
        
        self.stage = omni.usd.get_context().get_stage()
        self.materials = list(self.templates.keys()) #iterable of available materials for randomisation
        self.materials_in_scene = [] #stores the paths of materials that have been created in the scene 




    def reset(self):
        """Call this at the start of every data_generator_loop iteration"""
        for mat in self.materials_in_scene:
            prims.delete_prim(mat)
        self.materials_in_scene = []
        self.active_scene_materials = []
    
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
            while True:
                template_key = random.choice(self.materials)
                if "wood" not in template_key:
                    break
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
        if mat_prim_path is not None and mat_name != "Plastic_Standardized_Surface_Finish_V15":
            carb.log_info(f"[MaterialManager] Created material {mat_name} at {mat_prim_path}")
            self.materials_in_scene.append(mat_prim_path)

        # 2. Smart Wait for Compilation
        # We check if the shader is in the registry. 
        # We search by filename because 'file:///...' might differ from registry's internal path.
        target_filename = os.path.basename(mdl_url)
        registry = Sdr.Registry()
        
        # Fast check: is it already there?
        found = True
        mat_prim = self.stage.GetPrimAtPath(mat_prim_path)
        if not mat_prim:
            found = False
            carb.log_warn(f"Material {mat_name} not found. Skipping randomization.")

        shader_path = mat_prim_path + "/Shader"
        shader_prim = self.stage.GetPrimAtPath(shader_path)
        if not shader_prim:
            found = False
            carb.log_warn(f"Shader {mat_name} not found. Skipping randomization.")

        if "randomise" in template_data:
            shader = UsdShade.Shader(shader_prim)
            
            for param_key, value_data in template_data["randomise"].items():
                
                if param_key=="plastic_color":
                    h = random.uniform(value_data['h'][0], value_data['h'][1])
                    s = random.uniform(value_data['s'][0], value_data['s'][1])
                    v = random.uniform(value_data['v'][0], value_data['v'][1])
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    
                    # Create as Color3f
                    shader.CreateInput("plastic_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
                    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(5.0, 5.0))

                elif isinstance(value_data, list) and len(value_data) == 2:
                    val = random.uniform(value_data[0], value_data[1])
                    
                    # Create as Float
                    shader.CreateInput(param_key, Sdf.ValueTypeNames.Float).Set(val)

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
        Binds a material to a prim with 'StrongerThanDescendants' strength.
        This forces the binding even if the child Mesh has its own material.
        """
        if prim_path is None:
            raise ValueError("prim_path must be specified")


        if mat_prim_path is None:
            if not self.materials_in_scene:
                carb.log_warn("No materials created yet.")
                return
            mat_prim_path = random.choice(self.materials_in_scene)


        prim = self.stage.GetPrimAtPath(prim_path)
        material = UsdShade.Material(self.stage.GetPrimAtPath(mat_prim_path))

        if not prim.IsValid() or not material.GetPrim().IsValid():
            carb.log_warn(f"Invalid prim or material: {prim_path} / {mat_prim_path}")
            return


        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        

        binding_api.Bind(
            material, 
            bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )
        

        print(f"[MaterialManager] Bound {mat_prim_path} to {prim_path} (StrongerThanDescendants)")
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
        """
        Checks if the shader is loaded. 
        Note: 'filename' passed here is usually 'MaterialName.mdl'.
        We must check for 'MaterialName' without the extension.
        """
        # 1. Strip extension to get the likely Sdr identifier part
        # e.g. "Copper_Scratched.mdl" -> "Copper_Scratched"
        base_name = os.path.splitext(filename)[0]
        
        # 2. Iterate and check
        # Sdr identifiers are often full URIs like:
        # '::vMaterials_2::Metal::Copper_Scratched::Copper_Scratched'
        for uri in registry.GetNodeIdentifiers():
            if base_name in uri:
                return True
                
        return False