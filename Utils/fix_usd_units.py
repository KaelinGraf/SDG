from pxr import Usd, UsdGeom, Sdf

# Path to your bin USD file
usd_path = "../Meshes/bin_1.usd" 

def sanitize_usd(file_path):
    # Open the layer
    stage = Usd.Stage.Open(file_path)
    
    # 1. Fix Meters Per Unit (MPU)
    # Isaac Sim uses 1.0 (1 unit = 1 meter). 
    # If your file was CM, this might be 0.01. We force it to 1.0 to match the Sim.
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    
    # 2. Fix Up Axis (Isaac Sim uses Z-Up)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Save the changes back to the file
    stage.GetRootLayer().Save()
    print(f"SUCCESS: Updated {file_path} to MetersPerUnit=1.0 and UpAxis=Z")

if __name__ == "__main__":
    sanitize_usd(usd_path)