import numpy as np
import open3d
import pyvista as pv
import copy
from utils import *

if __name__ == "__main__":
    # read file
    filename = "STF_Si_L768_x0.85_Th85.5_D20_N25165824_1660895694"
    grid, deposited, params = loadSparse('structures/' + filename + '.npz')
    print("Maxima: {}".format(np.max(grid, axis=0)[:3]))
    
    
    # colors
    silver = [0.753, 0.753, 0.753]#{'color': [0.753, 0.753, 0.753], 'material': 
    silicon = [0.600, 0.460, 0.357]
    c_silicon = [0.5, 0.5, 0.5]
    gold = [212/255, 175/255, 55/255]
    colors = [[0,0,0], silicon, silver, gold]
    
    ''' Open3d '''
    
    #pcd = open3d.geometry.PointCloud()
    #for i in range(grid.shape[0]):
    #    pcd.points.append(grid[i,:3])
    #    pcd.colors.append(colors[grid[i,3]])
    #print(pcd)
    
    #open3d.visualization.draw([pcd])
    
    ''' PyVista '''
    
    si_grid = []
    ag_grid = []
    
    for i in range(grid.shape[0]):
        if grid[i,3] == 1:
            si_grid.append(grid[i,:3])
        elif grid[i,3] == 2:
            ag_grid.append(grid[i,:3])
    
    #ag_pcd = pv.PolyData(ag_grid[:,:3])
    #ag_pcd['species'] = ag_grid[:,3]
    #ag_pcd.compute_normals()
    
    #si_pcd = pv.PolyData(si_grid[:,:3])
    #si_pcd['species'] = si_grid[:,3]
    #si_pcd.compute_normals()
    
    pcd = pv.PolyData(grid[:,:3])
    pcd['species'] = grid[:,3]
    pcd.compute_normals()
    
    pcd.plot(eye_dome_lighting=True)