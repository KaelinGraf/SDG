from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
}) 

import carb
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core import World


usd_path="/home/kaelin/Desktop/custom_usds/warehouse_ur12e.usd"
open_stage(usd_path)
world = World()
world.reset()

while simulation_app.is_running():
    # This triggers the renderer and physics step
    simulation_app.update() 