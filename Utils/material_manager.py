#methods for templating, randomising and applying realistic materials to USD prims



class MaterialManager:
    def __init__(self):
        self.templates={
            "aluminium_anodized":{
                "mdl_path":"/home/kaelin/BinPicking/SDG/Materials/Aluminum_Anodized.mdl",
                'name':'aluminium_anodized',
                'non_visual':('aluminum,none,emissive'),
                'randomise':{
                    'specular':(0.0,1.0),
                    'roughness':(0.0,1.0),
                    'metallic':(0.0,1.0)
                }

            },
            'aluminium_brushed':{
                "mdl_path":"/home/kaelin/BinPicking/SDG/Materials/Aluminum_Brushed.mdl",
                'name':'aluminium_brushed',
                'non_visual':('aluminum,none,emissive'),
                'randomise':{
                    'specular':(0.0,1.0),
                    'roughness':(0.0,1.0),
                    'metallic':(0.0,1.0)
                }

            },
            'aluminum_cast':{
                'mdl_path':'/home/kaelin/BinPicking/SDG/Materials/Aluminum_Cast.mdl',
                'name':'aluminum_cast',
                'non_visual':('aluminum,none,emissive'),
                'randomise':{
                    'specular':(0.0,1.0),
                    'roughness':(0.0,1.0),
                    'metallic':(0.0,1.0)
                }

            },
            'aluminum_scratched':{
                'mdl_path':'/home/kaelin/BinPicking/SDG/Materials/Aluminum_Scratched.mdl',
                'name':'aluminum_scratched',
                'non_visual':('aluminum,none,emissive'),
                'randomise':{
                    'specular':(0.0,1.0),
                    'roughness':(0.0,1.0),
                    'metallic':(0.0,1.0)
                }

            },
            'brass':{
                'mdl_path':"/home/kaelin/BinPicking/SDG/Materials/Brass.mdl",
                'name':'brass',
                'non_visual':('brass,none,emissive'),
                'randomise':{
                    'specular':(0.0,1.0),
                    'roughness':(0.0,1.0),
                    'metallic':(0.0,1.0)
                }

            },
            'brushed_antique_copper':{
                'mdl_path':'/home/kaelin/BinPicking/SDG/Materials/Brushed_Antique_Copper.mdl',
                'name':'brushed_antique_copper',
                'non_visual':('bronze,none,emissive'),
                'randomise':{
                    'specular':(0.0,1.0),
                    'roughness':(0.0,1.0),
                    'metallic':(0.0,1.0)
                }

            }
            
        }
