import numpy as np
import pyvista as pv
import open3d
import scipy.ndimage
import scipy.stats
import copy
from utils import *
from matplotlib import pyplot as plt
from matplotlib import cm
import tensorflow as tf

#si_material = vis.Material('defaultLitSSR')
#si_material.scalar_properties['m']

def get_bounding_box(array, val):
    indices = np.argwhere(array == val)
    ll = (np.min(indices[:,0]), np.min(indices[:,1]), np.min(indices[:,2]))
    ur = (np.max(indices[:,0]), np.max(indices[:,1]), np.max(indices[:,2]))
    return ll, ur

def hollow(array, dist=5, six_conn=True):
    if dist == 0:
        return array
    # get bounding box
    indices = np.argwhere(array == 1)
    ll = (np.min(indices[:,0]), np.min(indices[:,1]), np.min(indices[:,2]))
    ur = (np.max(indices[:,0]), np.max(indices[:,1]), np.max(indices[:,2]))
    print("Bounding box: {}, {}".format(ll, ur))
    
    c = np.zeros(array.shape)
    # find los for all points to edges of bounding box
    for n in indices:
        shadowed = False
        # check boundaries
        if n[2] == ll[2] or n[2] == ur[2] or n[1] == ll[1] or n[1] == ur[1] or n[0] == ll[0] or n[0] == ur[0]:
            c[n[0], n[1], n[2]] = 1
            continue
        
        # -x
        for i in range(n[2]-dist, n[2]):
            if i < ll[2]:
                continue
            if array[n[0], n[1], i] == 1:
                shadowed = True
                break
        if not shadowed:
            c[n[0], n[1], n[2]] = 1
            continue
        shadowed = False
                
        # +x
        for i in range(n[2]+1, n[2]+dist+1):
            if i > ur[2]:
                break
            if array[n[0], n[1], i] == 1:
                shadowed = True
                break
        if not shadowed:
            c[n[0], n[1], n[2]] = 1
            continue
        shadowed = False
        
        # -y
        for j in range(n[1]-dist, n[1]):
            if j < ll[1]:
                continue
            if array[n[0], j, n[2]] == 1:
                shadowed = True
                break
        if not shadowed:
            c[n[0], n[1], n[2]] = 1
            continue
        shadowed = False
                
        # +y
        for j in range(n[1]+1, n[1]+dist+1):
            if j > ur[1]:
                break
            if array[n[0], j, n[2]] == 1:
                shadowed = True
                break
        if not shadowed:
            c[n[0], n[1], n[2]] = 1
            continue
        shadowed = False
        
        # -z
        for k in range(n[0]-dist, n[0]):
            if k < ll[0]:
                continue
            if array[k, n[1], n[2]] == 1:
                shadowed = True
                break
        if not shadowed:
            c[n[0], n[1], n[2]] = 1
            continue
        shadowed = False
                
        # +z
        for k in range(n[0]+1, n[0]+dist+1):
            if k > ur[0]:
                break
            if array[k, n[1], n[2]] == 1:
                shadowed = True
                break
        if not shadowed:
            c[n[0], n[1], n[2]] = 1
            continue
        shadowed = False
    return c

def interior_points(pcd, dist=5):
    if dist == 0:
        return array
    # get bounding box
    ll = (np.min(np.array(pcd.points)[:,0]), np.min(np.array(pcd.points)[:,1]), np.min(np.array(pcd.points)[:,2]))
    ur = (np.max(np.array(pcd.points)[:,0]), np.max(np.array(pcd.points)[:,1]), np.max(np.array(pcd.points)[:,2]))
    #print("Bounding box: {}, {}".format(ll, ur))
    
    interior = []
    # find los for all points to edges of bounding box
    for j in range(len(pcd.points)):
        n = pcd.points[j]
        
        # check boundaries
        if n[2] == ll[2] or n[2] == ur[2] or n[1] == ll[1] or n[1] == ur[1] or n[0] == ll[0] or n[0] == ur[0]:
            continue
            
        #print(n)
        shadowedx = False
        shadowedy = False
        shadowedz = False
        
        # negative
        for i in range(-dist, 0):
            # -x
            if n[0] + i >= ll[0]:
                #print("in x bounds")
                if pcd.points.count(np.array([n[0] + i, n[1], n[2]])) > 0:
                    shadowedx = True
            # -y
            if n[1] + i >= ll[1]:
                if pcd.points.count(np.array([n[0], n[1] + i, n[2]])) > 0:
                    shadowedy = True
            # -z
            if n[2] + i >= ll[2]:
                if pcd.points.count(np.array([n[0], n[1], n[2] + i])) > 0:
                    shadowedz = True
        if not (shadowedx and shadowedy and shadowedz):
            continue
        #print("Shadowed in - direction")
        shadowedx = False
        shadowedy = False
        shadowedz = False
                
        # positive
        for i in range(1, dist + 1):
            # +x
            if n[0] + i <= ur[0]:
                if pcd.points.count(np.array([n[0] + i, n[1], n[2]])) > 0:
                    shadowedx = True
            # +y
            if n[1] + i <= ur[1]:
                if pcd.points.count(np.array([n[0], n[1] + i, n[2]])) > 0:
                    shadowedy = True
            # +z
            if n[2] + i <= ur[2]:
                if pcd.points.count(np.array([n[0], n[1], n[2] + i])) > 0:
                    shadowedz = True
        if shadowedx and shadowedy and shadowedz:
            interior.append(j)
    return interior

def smooth_normals(pcd):
    s3 = np.sqrt(3)
    # get points and flags ready
    num = len(pcd.points)
    flagged = np.zeros((num))
    point_array = np.asarray(pcd.points)
    f = np.argmax(point_array, axis=0)[0]
    flagged[f] = 1
    l = list(pcd.points)
    
    # create tree
    tree = open3d.geometry.KDTreeFlann(pcd)
    
    # init stack
    point_stack = []
    point_stack.append(f)
    while len(point_stack) > 0:
        point = point_stack.pop()
        #print(point)
        # nearest neighbors
        [k, idx, _] = tree.search_radius_vector_3d(pcd.points[point], 1)
        for i in idx[1:]:
            #print(i)
            # operate on and flag index as processed
            if flagged[i] == 0:
                # acute |a+b| > |a-b|
                if np.linalg.norm(pcd.normals[point] + pcd.normals[i]) < np.linalg.norm(pcd.normals[point] - pcd.normals[i]):
                    pcd.normals[i] = -pcd.normals[i]
                flagged[i] = 1
                point_stack.append(i)
    return


def calc_normals(points):
    o = np.mean(points, axis=0)
    #print("Center: {}".format(o))
    v = points - o
    v = v / np.linalg.norm(v, axis=1)[:, None]
    return v

def calc_normals(pcd):
    # need to have normals instantiated
    if len(pcd.normals) == 0:
        pcd.normals = open3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))
        
    o = np.mean(np.asarray(pcd.points), axis=0)
    v = np.asarray(pcd.points) - o
    pcd.normals = open3d.utility.Vector3dVector(v / np.linalg.norm(v, axis=1)[:, None])
    return v

def clean_porous_normals(points, normals, break_on_flip=False):
    o = np.mean(points, axis=0)
    m = np.max(points, axis=0)
    grid = matrixFromSparse(points)
    normalsc = copy.deepcopy(normals)
    for i in range(normals.shape[0]):
        n = normals[i,:]/np.max(np.abs(normals[i,:]))
        p = points[i,:3]
        v = p + n
        while v[2] < grid.shape[2] and v[2] > 0 and v[1] < grid.shape[1] and v[1] > 0 and v[0] < grid.shape[0] and v[0] > 0:
            if grid[int(v[0]), int(v[1]), int(v[2])] > 0:
                normalsc[i] = -normalsc[i]
                if break_on_flip:
                    break
            v += n
    return normalsc

def clean_porous_normals(pcd, grid=None, break_on_flip=True, verbose=False):
    if len(pcd.normals) == 0:
        raise TypeError("Normals must be given before they can be cleaned")
    o = np.mean(pcd.points, axis=0)
    m = np.max(pcd.points, axis=0)
    points = np.asarray(pcd.points)
    if grid is None:
        grid = np.round((points)).astype(np.int32)
        grid = np.hstack([grid, np.zeros((grid.shape[0], 1), dtype=np.int32)+1])
        grid = matrixFromSparse(grid)
    if verbose:
        print("Grid shape: {}".format(grid.shape))
    normals = np.asarray(pcd.normals)
    for i in range(normals.shape[0]):
        n = normals[i,:]/np.max(np.abs(normals[i,:]))/2
        p = points[i,:3]
        v = p + n
        if verbose:
            print("{}: {} + {} = {}".format(i, p, n, v))
        iv = [int(np.round(v[0])), int(np.round(v[1])), int(np.round(v[2]))]
        while iv[2] < grid.shape[0] and iv[2] >= 0 and iv[1] < grid.shape[1] and iv[1] >= 0 and iv[0] < grid.shape[2] and iv[0] >= 0:
            if verbose:
                print("\t{}: {}".format(iv, grid[iv[2], iv[1], iv[0]]))
            if grid[iv[2], iv[1], iv[0]] > 0 and not (iv[2] == p[2] and iv[1] == p[1] and iv[0] == p[0]):
                #print('flip')
                pcd.normals[i] = [-pcd.normals[i][0], -pcd.normals[i][1], -pcd.normals[i][2]]
                if break_on_flip:
                    if verbose:
                        print('break')
                    break
            v += n
            iv = [int(np.round(v[0])), int(np.round(v[1])), int(np.round(v[2]))]
    return

def calc_local_normals(pcd):
    # need to have normals instantiated
    if len(pcd.normals) == 0:
        pcd.normals = open3d.utility.Vector3dVector(np.zeros((len(pcd.points), 3)))
                                                    
    # create tree
    tree = open3d.geometry.KDTreeFlann(pcd)
    for i in range(len(pcd.points)):
        # get all neighbors (equivalent to nearest neighbors within unit cube diagonal)
        [k, idx, _] = tree.search_radius_vector_3d(pcd.points[i], np.sqrt(3))
        local_points = np.asarray(pcd.points)[idx, :]
        o = np.mean(local_points, axis=0)
        v = np.linalg.svd((local_points - o).T)[0] # leftmost vector
        pcd.normals[i] = v[:,-1]
    return
    
    

def remove_unanchored(grid, grid_fill):
    regions, number_of_components = scipy.ndimage.label(grid_fill)
    inv_regions = np.where(regions == 1, 1, 0)
    return grid * inv_regions


if __name__ == "__main__":
    silver = [0.753, 0.753, 0.753]#{'color': [0.753, 0.753, 0.753], 'material': 
    a_si = [0.600, 0.460, 0.357]
    gold = [212/255, 175/255, 55/255]
    
    filename = 'STF_Si_Ag_L768_x1.0_Th85.5_D60_N9437184_1660238350'
    
    grid, deposited, params = loadSparse('structures//' + filename + '.npz')
    print(params)
    
    grid_full = matrixFromSparse(grid)
    grid_si = np.where(grid_full==1, 1, 0)
    grid_ag = np.where(grid_full==2, 1, 0)
    
    ''' Open3D with separated components (no hanging)'''
    start = 6

    # filled si, ag, and all
    grid_si_fill = scipy.ndimage.binary_fill_holes(grid_si).astype('int32')[start:,:,:]
    grid_ag_fill = scipy.ndimage.binary_fill_holes(np.tile(grid_ag, (1,1,1))).astype('int64')[start:,:,:]
    grid_fill = scipy.ndimage.binary_fill_holes(np.tile(np.where(grid_full > 0, 1, 0), (1,1,1)))
    substrate = copy.deepcopy(grid_si)[:start,:,:]

    # disconnect components
    regions, number_of_components = scipy.ndimage.label(grid_fill)
    inv_regions = np.where(regions[start:,:,:] == 1, 1, 0)
    grid_si_fill *= inv_regions
    grid_ag_fill *= inv_regions
    inv_regions = np.where(regions[:start,:,:] == 1, 1, 0)
    substrate *= inv_regions
    print("Disconnected floating components.")

    # hollow out both materials
    si_hollow = hollow(grid_si_fill)
    ag_hollow = hollow(grid_ag_fill)
    substrate_hollow = hollow(substrate)
    print("Hollowed components.")

    # prepare small sphere
    #sphere_mesh = open3d.geometry.TriangleMesh.create_sphere()
    #mesh.compute_vertex_normals()
    #sphere_pcd = sphere_mesh.sample_points_uniformly(number_of_points=500)
    #print(sphere_pcd)

    # separate regions and create point clouds
    ag_comps = []
    regions, number_of_components = scipy.ndimage.label(ag_hollow, structure=np.ones((3,3,3)))
    print("{} Ag regions:".format(number_of_components))
    j = 0
    for i in range(number_of_components):
        reg = np.where(regions == i+1, 1, 0)
        mat = sparseFromMatrix(reg)
        
        if np.sum(mat[:,3]) < 5:
            continue
            #ag_comps[j].points = open3d.utility.Vector3dVector(np.asarray(sphere_pcd.points) + np.mean(mat[:,3], axis=0))
            #calc_normals(ag_comps[j])
        else:
            ag_comps.append(open3d.geometry.PointCloud())
            ag_comps[j].points = open3d.utility.Vector3dVector(mat[:,:3] + [0, 0, start-1])
            calc_local_normals(ag_comps[j])
            clean_porous_normals(ag_comps[j], reg)
            ag_comps[j].remove_radius_outlier(3, 2)
        ag_comps[j].paint_uniform_color(silver)
        j += 1
        #open3d.visualization.draw_geometries([ag_comps[i]])
    print("\t{} large enough to mesh.".format(len(ag_comps)))
    si_comps = []
    regions, number_of_components = scipy.ndimage.label(si_hollow, structure=np.ones((3,3,3)))
    print("{} Si regions:".format(number_of_components))
    j = 0
    for i in range(number_of_components):
        reg = np.where(regions == i+1, 1, 0)
        mat = sparseFromMatrix(reg)
        if np.sum(mat[:,3]) < 20:
            continue
        si_comps.append(open3d.geometry.PointCloud())
        si_comps[j].points = open3d.utility.Vector3dVector(mat[:,:3] + [0, 0, start-1])
        calc_local_normals(si_comps[j])
        clean_porous_normals(si_comps[j], reg)
        #si_comps[j].random_down_sample(0.2)
        #normals = np.asarray(si_comps[j].normals)#calc_normals(mat[:,:3])
        #si_comps[j].normals = open3d.utility.Vector3dVector(clean_porous_normals(mat, normals, True))
        #si_comps[j].remove_radius_outlier(3, 2)
        #si_comps[j].remove_radius_outlier(3, 2)
        for i in range(len(si_comps[j].points)):
            if si_comps[j].points[i][2] == start-1:
                si_comps[j].normals[i] = [0, 0, -1]
        si_comps[j].paint_uniform_color(a_si)
        j += 1
        #open3d.visualization.draw_geometries([si_comps[i]])
    print("\t{} large enough to mesh.".format(len(si_comps)))
    mat = sparseFromMatrix(substrate_hollow)
    #si_comps.append(open3d.geometry.PointCloud())
    #si_comps[-1].points = open3d.utility.Vector3dVector(mat[:,:3])
    #si_comps[-1].normals = open3d.utility.Vector3dVector(np.zeros((len(si_comps[-1].points), 3)))
    #for i in range(len(si_comps[-1].points)):
    #    if si_comps[-1].points[i][2] == 0:
    #        si_comps[-1].normals[i] = [0, 0, -1]
    #    else:
    #        si_comps[-1].normals[i] = [0, 0, 1]
    #si_comps[-1].paint_uniform_color(a_si)
    #print("Substrate cloud created.")
    print("Created point clouds.")
    
    
    # create meshes from the point clouds
    #ev = open3d.visualization.ExternalVisualizer()
    #import tensorflow as tf

    import time as time

    ag_meshes = [None]*len(ag_comps)
    start = time.time()
    endcube = 0
    endhollow = 0
    endrecon = 0
    for i in range(len(ag_comps)):
        cloud = ag_comps[i]
        
        # create cubes
        startcube = time.time()
        cs = open3d.geometry.TriangleMesh()
        for p in cloud.points:
            c = open3d.geometry.TriangleMesh.create_box()
            c.compute_vertex_normals()
            cs += c.translate(np.asarray(p))
        cs.compute_vertex_normals()
        endcube += time.time() - startcube
        
        # cubes to hollow point cloud of only surface
        starthollow = time.time()
        pcd = open3d.geometry.PointCloud()
        pcd.points = cs.vertices
        pcd.normals = cs.vertex_normals
        interiors = interior_points(pcd)
        pcd = pcd.select_by_index(interiors, invert=True)
        pcd = pcd.voxel_down_sample(0.5)
        endhollow += time.time() - starthollow
        
        # construct surface
        startrecon = time.time()
        try:
            #ag_meshes[i] = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0)
            ag_meshes[i] = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, width=1.5, scale=1.1, linear_fit=False)[0]
        except:
            ag_meshes[i] = open3d.geometry.TriangleMesh()
        endrecon += time.time() - startrecon
        
        ag_meshes[i].compute_vertex_normals()
        ag_meshes[i].merge_close_vertices(1)
        ag_meshes[i].paint_uniform_color(silver)
        #ag_meshes[i] = cloud.compute_convex_hull(True)[0]
        #ag_meshes[i] = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, open3d.utility.DoubleVector([3**0.5/4, 3**0.5/2]))
        #ag_meshes[i] = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=8, width=0, scale=1.1, linear_fit=False)[0]
        #ag_meshes[i].compute_vertex_normals()
        #ag_meshes[i].merge_close_vertices(1)
        #dec_mesh = ag_meshes[i].simplify_quadric_decimation(100000)
        #if not ag_meshes[i].is_watertight():
        #    ag_meshes[i].paint_uniform_color([1, 0.5, 0.5])
        #else:
        #    ag_meshes[i].paint_uniform_color([silver])
        #ag_meshes[i] = open3d.t.TriangleMesh.from_legacy(ag_meshes[i])
        #ag_meshes[i].material = open3d.visualization.Material('defaultLit')
        #ag_meshes[i].material.scalar_properties['reflectance'] = 0.15
        #ev.set(ag_meshes[i])
    print("Cube addition took {:.3f} seconds\nHollowing took {:.3f} seconds\nReconstruction took {:.3f} seconds".format(endcube, endhollow, endrecon))
    print("Silver meshed.")
    si_meshes = [None]*len(si_comps)
    i = 0
    for i in range(len(si_comps)):
        cloud = si_comps[i]
        
        # create cubes
        #cs = open3d.geometry.TriangleMesh()
        #for p in cloud.points:
        #    c = open3d.geometry.TriangleMesh.create_box()
        #    c.compute_vertex_normals()
        #    cs += c.translate(np.asarray(p))
        #cs.compute_vertex_normals()
        
        # cubes to hollow point cloud of only surface
        #pcd = open3d.geometry.PointCloud()
        #pcd.points = cs.vertices
        #pcd.normals = cs.vertex_normals
        #interiors = interior_points(pcd)
        #pcd = pcd.select_by_index(interiors, invert=True)
        #pcd = pcd.voxel_down_sample(0.5)
        #try:
            #ag_meshes[i] = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0)
        #    si_meshes[i] = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0.5, scale=1.1, linear_fit=False)[0]
        #except:
        #    si_meshes[i] = open3d.geometry.TriangleMesh()
        #si_meshes[i].compute_vertex_normals()
        #si_meshes[i].merge_close_vertices(1)
        
        #si_meshes[i] = open3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(cloud, open3d.utility.DoubleVector([3**0.5/8, 3**0.5]))
        si_meshes[i] = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=8, width=0, scale=1.1, linear_fit=True)[0]
        #si_meshes[i] = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha=10)
        si_meshes[i].merge_close_vertices(1)
        #dec_mesh = mesh.simplify_quadric_decimation(100000)
        si_meshes[i].compute_vertex_normals()
        si_meshes[i].paint_uniform_color(a_si)
        #ev.set(si_meshes[i])
    print("Silicon meshed.")
    open3d.visualization.draw_geometries([*si_meshes, *ag_meshes])
    
    
    # combine clouds and meshes
    MasterSiCloud = open3d.geometry.PointCloud()
    MasterAgCloud = open3d.geometry.PointCloud()
    MasterSiMesh = open3d.geometry.TriangleMesh()
    MasterAgMesh = open3d.geometry.TriangleMesh()
    for c in si_comps:
        MasterSiCloud += c
    for c in ag_comps:
        MasterAgCloud += c
    for c in si_meshes:
        MasterSiMesh += c
    for c in ag_meshes:
        MasterAgMesh += c
    MasterCloud = MasterSiCloud + MasterAgCloud
    MasterMesh = MasterSiMesh + MasterAgMesh

    # save clouds and meshes
    open3d.io.write_point_cloud('models/' + filename + '1.pcd', MasterCloud)
    open3d.io.write_triangle_mesh('models/' + filename + '1.ply', MasterMesh)
    open3d.io.write_point_cloud('models/' + filename + '1_Si.pcd', MasterSiCloud)
    open3d.io.write_triangle_mesh('models/' + filename + '1_Si.ply', MasterSiMesh)
    open3d.io.write_point_cloud('models/' + filename + '1_Ag.pcd', MasterAgCloud)
    open3d.io.write_triangle_mesh('models/' + filename + '1_Ag.ply', MasterAgMesh)