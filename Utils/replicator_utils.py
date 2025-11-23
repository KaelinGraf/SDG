#Helper functions for scene_building that relate to Omniverse Replicator. 
#This includes setting up/randomising sensors, defining materials, and defining annotators/writers

import isaacsim.zivid as zivid_sim

class RepCam:
    def __init__(self):
        """
        Use the zivid IsaacSim api to generate a zivid camera, and create a replicator camera to move with it (to capture annotation data).
        The camera is moved n times during the capturing of a scene (after the phyiscs steps have been completed) to generate N unique datapoints per scene.
        This approach is preferable over spawning N cameras, due to the reduced resource usage. 
        The camera is not destroyed between scenes as to minimise the chance of memory leaks.
        """
        self.zivid_camera = zivid_sim.camera.ZividCamera(
            prim_path="/World/ZividCamera",
            model_name = zivid_sim.camera.models.ZividCameraModelName.ZIVID_2_PLUS_MR60
        )