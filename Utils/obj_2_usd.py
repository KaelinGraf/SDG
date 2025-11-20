#batch converts OBJ files to USD format for Isaac Sim usage
#does not handle materials or textures, only geometry conversion
#designed to handle batch conversion of multiple OBJ files in a directory, using a JSON config with a "mesh_filepath" entry for each object
#saves converted USD files in the same directory as the source OBJ files, with the same base filename, but with a .usd extension, and adds
#a usd_filepath entry to the object config for later use in scene building

#kaelin graf-ogilvie 2025
ZIVID_EXT_PATH = "/home/kaelin/zivid-isaac-sim/source"

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
})
import omni
from isaacsim.core.utils.extensions import enable_extension
enable_extension("omni.kit.asset_converter")


import os
import json
import argparse
import asyncio

class Obj2UsdConverter:
    def __init__(self, config_path="./Config/objects.json"):
        self.config_path = config_path
        self.objects_config = self.load_config()
    

    def setup_context(self):
        #add any context setup here if needed
        self.context.ignore_camera = True
        self.context.ignore_light = True
    
    def load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        else:
            with open(self.config_path, 'r') as f:
                return json.load(f)
    
    def convert_all(self):
        for obj_name, obj_info in self.objects_config.items():
            obj_filepath = obj_info.get("mesh_filepath", None) #get the OBJ file path, must be specified under "mesh_filepath"
            if obj_filepath and obj_filepath.lower().endswith('.obj'):
                usd_filepath = self.convert_obj_to_usd(obj_filepath)
                if usd_filepath:
                    self.objects_config[obj_name]["usd_filepath"] = usd_filepath
                    print(f"Converted {obj_name}: {obj_filepath} -> {usd_filepath}")
                else:
                    print(f"Failed to convert {obj_name}: {obj_filepath}")
            else:
                print(f"No valid OBJ filepath for {obj_name}")
        self.save_updated_config()
    
    def convert_one(self, obj_name):
        obj_info = self.objects_config.get(obj_name, None)
        if obj_info:
            obj_filepath = obj_info.get("mesh_filepath", None)
            if obj_filepath and obj_filepath.lower().endswith('.obj'):
                status = asyncio.get_event_loop().run_until_complete(
                        convert_obj_to_usd(obj_filepath)
                    )
                if status:
                    self.objects_config[obj_name]["usd_filepath"] = self.usd_filepath
                    print(f"Converted {obj_name}: {obj_filepath} -> {self.usd_filepath}")
                    self.save_updated_config()
                else:
                    print(f"Failed to convert {obj_name}: {obj_filepath}")
            else:
                print(f"No valid OBJ filepath for {obj_name}")
        else:
            print(f"Object {obj_name} not found in config")
    
    async def convert_obj_to_usd(self, obj_filepath):
        import omni.kit.asset_converter as asset_converter
        self.converter = asset_converter.get_instance()
        self.context = asset_converter.AssetConverterContext()
        def progress_callback(progress, total_steps):
            pass
        self.usd_filepath = os.path.splitext(obj_filepath)[0] + '.usd' #replace .obj with .usd
        task = self.converter.create_converter_task(obj_filepath,self.usd_filepath,progress_callback,self.context)
        while True:
            success = await task.wait_until_finished()
            if not success:
                await asyncio.sleep(0.1)
            else:
                break
            
            
    
    def save_updated_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.objects_config, f, indent="\t") #re-write with usd filepath 
        print(f"Updated config saved to {self.config_path}")
        
def main():
    parser = argparse.ArgumentParser(description="OBJ to USD Converter for Isaac Sim")
    parser.add_argument('--config_path', type=str, default="./Config/objects.json", help='Path to the objects configuration JSON file')
    parser.add_argument('--object_name', type=str, default=None, help='Name of a specific object to convert (if not provided, all objects will be converted)')
    args = parser.parse_args()
    
    converter = Obj2UsdConverter(args.config_path)
    if args.object_name:
        converter.convert_one(args.object_name)
    else:
        converter.convert_all()
        
    simulation_app.close()
    
if __name__ == "__main__":
    main()