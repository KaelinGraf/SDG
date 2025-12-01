from isaacsim import SimulationApp
import warp
simulation_app = SimulationApp({
    "headless": False,
}) 

import omni.client
import json

# Paste your list of URLs here to test them
urls_to_test = [
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Materials/Base/Metals/Aluminum_Anodized.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Materials/vMaterials_2/Metal/Aluminum_Brushed.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Materials/Base/Metals/Aluminum_Cast.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Materials/vMaterials_2/Metal/Aluminum_Scratched.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Materials/Base/Metals/Brass.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Materials/Base/Metals/Brushed_Antique_Copper.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Materials/Base/Metals/Copper.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Materials/Base/Plastics/Plastic_ABS.mdl",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Materials/vMaterials_2/Metal/Stainless_Steel.mdl",
]

print("--- Validating URLs ---")
for url in urls_to_test:
    result, entry = omni.client.stat(url)
    if result == omni.client.Result.OK:
        print(f"[OK] Found: {url}")
    else:
        print(f"[FAIL] {result.name}: {url}")