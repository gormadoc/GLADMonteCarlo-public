import numpy as np
import pyvista as pv
import copy
from utils import *

if __name__ == "__main__":
    # read file
    filename = ""
    agmesh = pv.read('models/' + 'STF_Si_Ag_L768_x1.0_Th85.5_D60_N9437184_1659679551_Ag' + '.ply')
    simesh = pv.read('models/' + 'STF_Si_Ag_L768_x1.0_Th85.5_D60_N9437184_1659679551_Si' + '.ply')
    print(agmesh)
    print(simesh)
    
    # colors
    silver = [0.753, 0.753, 0.753]#{'color': [0.753, 0.753, 0.753], 'material': 
    silicon = [0.600, 0.460, 0.357]
    c_silicon = [0.5, 0.5, 0.5]
    gold = [212/255, 175/255, 55/255]
    
    p = pv.Plotter()
    #cubemap = pv.examples.download_sky_box_cube_map()
    #p.add_actor(cubemap.to_skybox())
    #p.set_environment_texture(pv.examples.download_emoji_texture())
    p.set_background('black', top='white')
    light = pv.Light((384, -768, 200), (384,384,0), 'white', positional=True, intensity=0.5, show_actor=False)
    p.add_light(light)
    light = pv.Light((384, 384, 800), (384,384,0), 'white', positional=True, intensity=1, show_actor=False)
    p.add_light(light)
    p.add_mesh(agmesh, color=silver, lighting=None, ambient=0.2, smooth_shading=True, pbr=True, metallic=0.6, roughness=0.3, diffuse=0.5)
    p.add_mesh(simesh, color=silicon, lighting=None, ambient=0.2, smooth_shading=True, pbr=True, metallic=0.1, roughness=0.7, diffuse=0.5)
    p.enable_shadows()
    p.show()