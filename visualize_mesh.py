import numpy as np
import open3d
import copy
from utils import *

if __name__ == "__main__":
    # read file
    filename = "STF_Si_Ag_L768_Th85.5_D30_N9437184_1659178176"
    mesh = open3d.io.read_triangle_mesh('models/' + filename + '.ply')
    mesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
    print(mesh)
    
    # colors
    silver = [0.753, 0.753, 0.753]#{'color': [0.753, 0.753, 0.753], 'material': 
    silicon = [0.600, 0.460, 0.357]
    c_silicon = [0.5, 0.5, 0.5]
    gold = [212/255, 175/255, 55/255]
    
    # materials
    ag_mat = open3d.visualization.Material('defaultLit')
    si_mat = open3d.visualization.Material('defaultLit')
    #mesh.material = (ag_mat)
    #mesh.material.append(si_mat)
    print(ag_mat)
    print(ag_mat.scalar_properties)
    ag_mat.scalar_properties['reflectance'] = 0.15
    si_mat.scalar_properties['roughness'] = 0.15
    
    print(mesh.triangle.get_primary_key())
    
    # iterate over triangles
    #for i in range(len(mesh.triangle['colors'])):
    #    if mesh.triangle['colors'][i] == silver:
    #        mesh.triangle['material'][i] = 0
    #    else:
    #        mesh.triangle['material'][i] = 1
    #    ag_meshes[i] = open3d.t.TriangleMesh.from_legacy(ag_meshes[i])
    #    ag_meshes[i].material = open3d.visualization.Material('defaultLit')
    #    ag_meshes[i].material.scalar_properties['reflectance'] = 0.15
    open3d.visualization.draw([mesh])