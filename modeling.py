import open3d
import numpy as np
import scipy.ndimage
import tensorflow as tf
import copy
from utils import *

colors = {'a_si': [0.600, 0.460, 0.357], 
          'silver': [0.753, 0.753, 0.753], 
          'gold': [0.831, 0.686, 0.216]}
color_value_to_name = {1: 'a_si', 2: 'silver', 3: 'gold'}



def hollow(array, dist=5, six_conn=True):
    """
    Hollow a matrix and return as binary matrix (1 is a surface point, 0 is other),
        Parameters:
            array (3D ndarray): Dense matrix with important elements set to 1.
            dist (int): Local distance to determine whether each point is inside or outside the volume. (default=5)
            six_conn (bool): Check only parallel to axes if True; no current effect. (default=True)
        Returns:
            c (3D ndarray): Locally hollow binarized _array_ where 1 is surface points and everything else is 0.
    """
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


def get_border(sparse):
    # convert to a matrix (ids = relative position + 1)
    mini = np.min(sparse, axis=0).astype(np.int32)
    maxi = np.max(sparse, axis=0).astype(np.int32)

    z = np.zeros((maxi[0] - mini[0] + 2, maxi[1] - mini[1] + 2, maxi[2] - mini[2] + 2), dtype=np.int32)
    
    for k in range(sparse.shape[0]):
        l = (sparse[k,:] - mini + 1).astype(np.int32)
        try:
            z[l[0], l[1], l[2]] = 1
        except IndexError:
            print("Z: {}".format(z.shape))
            print("Mins: {}".format(mini))
            print("Maxes: {}".format(maxi))
            print("l: {}".format(l))
            raise IndexError()
    
    print(z)
    
    # flood fill
    regions, number_of_components = scipy.ndimage.label(z)
    
    # get label for an upper corner to be outside space
    space_lbl = regions[-1, 0, 0]
    
    # separate into space and inside
    regions = np.where(regions==space_lbl, 0, 1)
    
    # get edges
    edges = regions - scipy.ndimage.binary_erosion(regions)
    
    # make a new list of the edge pixels
    listo = []
    for k in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            for i in range(edges.shape[2]):
                if edges[k,j,i] == 1:
                    listo.append(np.array([k,j,i]) + mini[:3] - 1)
    return np.asarray(listo)

def get_border_indices(sparse):
    """
    Get a list of the indices of the surface points in a sparse matrix.
        Parameters:
            sparse (Nx>2 ndarray): Sparse matrix listing nonzero elements in 3D ndarray; each row is [x, y, z, ...].
        Returns:
            listo (list): List of the indices of surface points in sparse.
    """
    
    # convert to a matrix (ids = relative position + 1)
    mini = np.min(sparse, axis=0).astype(np.int32)
    maxi = np.max(sparse, axis=0).astype(np.int32)

    z = np.zeros((maxi[0] - mini[0] + 2, maxi[1] - mini[1] + 2, maxi[2] - mini[2] + 2), dtype=np.int32)
    
    for k in range(sparse.shape[0]):
        l = (sparse[k,:] - mini + 1).astype(np.int32)
        try:
            z[l[0], l[1], l[2]] = 1
        except IndexError:
            print("Z: {}".format(z.shape))
            print("Mins: {}".format(mini))
            print("Maxes: {}".format(maxi))
            print("l: {}".format(l))
            raise IndexError()
    
    # flood fill
    regions, number_of_components = scipy.ndimage.label(z)
    
    # get label for an upper corner to be outside space
    space_lbl = regions[-1, 0, 0]
    
    # separate into space and inside
    regions = np.where(regions==space_lbl, 0, 1)
    
    # get edges
    edges = regions - scipy.ndimage.binary_erosion(regions)#scipy.ndimage.sobel(regions)
    #edges = scipy.ndimage.binary_dilation(1 - regions) & regions
    
    # make a new list of the edge pixels
    listo = []
    for k in range(sparse.shape[0]):
        l = (sparse[k,:] - mini + 1).astype(np.int32)
        try:
            if edges[l[0], l[1], l[2]] > 0:
                listo.append(k)
        except IndexError:
            print("Z: {}".format(z.shape))
            print("Mins: {}".format(mini))
            print("Maxes: {}".format(maxi))
            print("l: {}".format(l))
            raise IndexError()
    return listo


def get_border_dense(dense):
    # pad matrix (ids = position + 1)
    z = np.pad(dense)
    
    # flood fill
    regions, number_of_components = scipy.ndimage.label(z)
    
    # get label for an upper corner to be outside space
    space_lbl = regions[-1, 0, 0]
    
    # separate into space and inside
    regions = np.where(regions==space_lbl, 0, 1)
    
    # get edges
    edges = regions - scipy.ndimage.binary_erosion(regions)
    
    # make a new list of the edge pixels
    listo = []
    for k in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            for i in range(edges.shape[2]):
                if edges[k,j,i] == 1:
                    listo.append(np.array([k,j,i]) - 1)
    return np.asarray(listo)


''' Mesh all of the grid together '''
def reconstructSiAgMeshesFull(filename):
    """
    Construct meshes from a point cloud assuming that the point cloud only contains Si or Si/Ag structures. Includes the substrate in the mesh.
        Parameters:
            filename (str): Filename of compressed sparse point cloud representing Si/Ag structures.
        Returns:
            si_mesh (open3d.geometry.TriangleMesh): Single mesh of Si components.
            ag_mesh (open3d.geometry.TriangleMesh): Single mesh of Ag components.
    """
    # load grid
    grid, deposited, params = loadSparse('structures//' + filename + '.npz')
    print(params)
    
    # create dense grid and separate
    grid_full = matrixFromSparse(grid)
    grid_si = np.where(grid_full==1, 1, 0)
    grid_ag = np.where(grid_full==2, 1, 0)
    
    # filled all
    grid_fill = scipy.ndimage.binary_fill_holes(np.tile(np.where(grid_full > 0, 1, 0), (1,1,1)))

    # disconnect components
    regions, number_of_components = scipy.ndimage.label(grid_fill)
    inv_regions = np.where(regions == 1, 1, 0)
    grid_si *= inv_regions
    grid_ag *= inv_regions
    print("Disconnected floating components.")
    
    # create point clouds from filled matrices
    mat = sparseFromMatrix(grid_ag)
    ag_pcd = open3d.geometry.PointCloud()
    ag_pcd.points = open3d.utility.Vector3dVector(mat[:,:3])
    ag_pcd.paint_uniform_color([0, 0.2, 1])
    mat = sparseFromMatrix(grid_si)
    si_pcd = open3d.geometry.PointCloud()
    si_pcd.points = open3d.utility.Vector3dVector(mat[:,:3])
    si_pcd.paint_uniform_color([0, 0.2, 1])
    print("Created point clouds.")
    
    # create all cubes for points in cloud
    cs = open3d.geometry.TriangleMesh()
    for p in ag_pcd.points:
        c = open3d.geometry.TriangleMesh.create_box()
        c.compute_vertex_normals()
        cs += c.translate(np.asarray(p))
    # combine identical vertices with different normals, clean up, and recalculate normals
    cs.merge_close_vertices(1)
    cs.remove_duplicated_triangles()
    cs.remove_degenerate_triangles()
    cs.compute_vertex_normals()
    # create pcd from cube vertices on surface
    pcd = open3d.geometry.PointCloud()
    keep = get_border_indices(np.asarray(cs.vertices))
    vert = []
    normals = []
    for j in range(len(keep)):
        vert.append(cs.vertices[keep[j]])
        normals.append(cs.vertex_normals[keep[j]])
    pcd.points = open3d.utility.Vector3dVector(vert)
    pcd.normals = open3d.utility.Vector3dVector(normals)
    # create surface if possible, compute normals, and color
    try:
        ag_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=2, scale=1.1, linear_fit=False)[0]
    except:
        ag_mesh = open3d.geometry.TriangleMesh()
    ag_mesh.compute_vertex_normals()
    ag_mesh.paint_uniform_color(colors['silver'])
    
    # do above again for silicon
    cs = open3d.geometry.TriangleMesh()
    for p in si_pcd.points:
        c = open3d.geometry.TriangleMesh.create_box()
        c.compute_vertex_normals()
        cs += c.translate(np.asarray(p))
    cs.merge_close_vertices(1)
    cs.remove_duplicated_triangles()
    cs.remove_degenerate_triangles()
    cs.compute_vertex_normals()
    pcd = open3d.geometry.PointCloud()
    keep = get_border_indices(np.asarray(cs.vertices))
    vert = []
    normals = []
    for j in range(len(keep)):
        vert.append(cs.vertices[keep[j]])
        normals.append(cs.vertex_normals[keep[j]])
    pcd.points = open3d.utility.Vector3dVector(vert)
    pcd.normals = open3d.utility.Vector3dVector(normals)
    try:
        si_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=2, scale=1.1, linear_fit=False)[0]
    except:
        si_mesh = open3d.geometry.TriangleMesh()
    si_mesh.compute_vertex_normals()
    si_mesh.paint_uniform_color(colors['a_si'])
    print("Created surface meshes.")
    return si_mesh, ag_mesh


''' Mesh each Si and Ag component separately '''
def reconstructSiAgMeshColumns(filename, start=6, color_by_column=False, remove_floating=True):
    """
    Construct meshes from a point cloud assuming that the point cloud only contains Si or Si/Ag structures. The substrate can be excluded along with the nucleation layer, floating components can be kept or discarded, and color differentiation by structure can be chosen.
        Parameters:
            filename (str): Filename of compressed sparse point cloud representing Si/Ag structures.
            start (int): The height/index to slice between structures and substrate/nucleation layer. This affects both where the meshes begin and where the columns are differentiated. (default=6)
            color_by_column (bool): False to color by material, True to color by column. (default=False)
            remove_floating (bool): False to keep floating parts, True to remove. (default=True)
        Returns:
            si_meshes (list of open3d.geometry.TriangleMesh): List of meshes of Si components.
            ag_meshes (list of open3d.geometry.TriangleMesh): List of meshes of Ag components.
    """
    # load grid
    grid, deposited, params = loadSparse('structures//' + filename + '.npz')
    print(params)

    # create dense grid and separate
    grid_full = matrixFromSparse(grid)
    grid_si = np.where(grid_full==1, 1, 0)
    grid_ag = np.where(grid_full==2, 1, 0)

    # filled all
    grid_fill = scipy.ndimage.binary_fill_holes(np.tile(np.where(grid_full > 0, 1, 0), (1,1,1)))
    substrate = copy.deepcopy(grid_si)[:start,:,:]
    grid_si = grid_si[start:,:,:]
    grid_ag = grid_ag[start:,:,:]

    # disconnect floating components if desired
    if remove_floating:
        regions, number_of_components = scipy.ndimage.label(grid_fill)
        inv_regions = np.where(regions[start:,:,:] == 1, 1, 0)
        grid_si *= inv_regions
        grid_ag *= inv_regions
        inv_regions = np.where(regions[:start,:,:] == 1, 1, 0)
        substrate *= inv_regions
        print("Disconnected floating components.")

    if color_by_column:
        columns, number_of_columns = scipy.ndimage.label(grid_fill[start:, :, :])
        cmap = np.random.rand(number_of_columns, 3)
        #cmap = cm.get_cmap(colormap, number_of_columns)
        #cmap = np.ndarray((number_of_columns+1, 4))
        #for i in range(0, number_of_columns+1):
        #    cmap[i] = cmapt(i/number_of_columns)

    # separate regions and create point clouds
    ag_meshes = []
    regions, number_of_components = scipy.ndimage.label(grid_ag, structure=np.ones((3,3,3)))
    if color_by_column:
        #columns, number_of_columns = scipy.ndimage.label(grid_fill[start:, :, :])
        cmap = np.random.rand(number_of_components, 3)
        #cmap = cm.get_cmap(colormap, number_of_columns)
        #cmap = np.ndarray((number_of_columns+1, 4))
        #for i in range(0, number_of_columns+1):
        #    cmap[i] = cmapt(i/number_of_columns)
    print("{} Ag regions:".format(number_of_components))
    n = 0
    x = 0
    points_left = 0
    for i in range(number_of_components):
        reg = np.where(regions == i+1, 1, 0)
        mat = sparseFromMatrix(reg)

        # create point clouds from filled matrices
        ag_pcd = open3d.geometry.PointCloud()
        ag_pcd.points = open3d.utility.Vector3dVector(mat[:,:3])
        ag_pcd.paint_uniform_color([0, 0.2, 1])

        # create all cubes for points in cloud
        cs = open3d.geometry.TriangleMesh()
        for p in ag_pcd.points:
            c = open3d.geometry.TriangleMesh.create_box()
            c.compute_vertex_normals()
            cs += c.translate(np.asarray(p))
        # combine identical vertices with different normals, clean up, and recalculate normals
        cs.merge_close_vertices(1)
        cs.remove_duplicated_triangles()
        cs.remove_degenerate_triangles()
        cs.compute_vertex_normals()
        # create pcd from cube vertices on surface
        pcd = open3d.geometry.PointCloud()
        keep = get_border_indices(np.asarray(cs.vertices))
        vert = []
        normals = []
        for j in range(len(keep)):
            vert.append(cs.vertices[keep[j]])
            normals.append(cs.vertex_normals[keep[j]])
        pcd.points = open3d.utility.Vector3dVector(vert)
        pcd.normals = open3d.utility.Vector3dVector(normals)
        # create surface if possible, compute normals, and color
        try:
            ag_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=2, scale=1.1, linear_fit=False)[0]
            n += 1
        except:
            ag_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0.5, scale=1.1, linear_fit=False)[0]
            #ag_mesh = pcd#open3d.geometry.TriangleMesh()
            x += 1
            points_left += len(pcd.points)
        ag_mesh.compute_vertex_normals()
        if color_by_column:
            ag_mesh.paint_uniform_color(cmap[i,:])
        else:
            ag_mesh.paint_uniform_color(colors['silver'])
        ag_meshes.append(ag_mesh)

    print("\t{} able to mesh; {} unable for {} points left out.".format(n, x, points_left))
    si_meshes = []
    regions, number_of_components = scipy.ndimage.label(grid_si, structure=np.ones((3,3,3)))
    if color_by_column:
        #columns, number_of_columns = scipy.ndimage.label(grid_fill[start:, :, :])
        cmap = np.random.rand(number_of_components, 3)
        #cmap = cm.get_cmap(colormap, number_of_columns)
        #cmap = np.ndarray((number_of_columns+1, 4))
        #for i in range(0, number_of_columns+1):
        #    cmap[i] = cmapt(i/number_of_columns)
    print("{} Si regions:".format(number_of_components))
    n = 0
    x = 0
    points_left = 0
    for i in range(number_of_components):
        reg = np.where(regions == i+1, 1, 0)
        mat = sparseFromMatrix(reg)

        si_pcd = open3d.geometry.PointCloud()
        si_pcd.points = open3d.utility.Vector3dVector(mat[:,:3])
        si_pcd.paint_uniform_color([0, 0.2, 1])

        cs = open3d.geometry.TriangleMesh()
        for p in si_pcd.points:
            c = open3d.geometry.TriangleMesh.create_box()
            c.compute_vertex_normals()
            cs += c.translate(np.asarray(p))
        cs.merge_close_vertices(1)
        cs.remove_duplicated_triangles()
        cs.remove_degenerate_triangles()
        cs.compute_vertex_normals()
        pcd = open3d.geometry.PointCloud()
        keep = get_border_indices(np.asarray(cs.vertices))
        vert = []
        normals = []
        for j in range(len(keep)):
            vert.append(cs.vertices[keep[j]])
            normals.append(cs.vertex_normals[keep[j]])
        pcd.points = open3d.utility.Vector3dVector(vert)
        pcd.normals = open3d.utility.Vector3dVector(normals)
        try:
            si_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=2, scale=1.1, linear_fit=False)[0]
            n += 1
        except:
            si_mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0.5, scale=1.1, linear_fit=False)[0]
            #si_mesh = pcd#open3d.geometry.TriangleMesh()
            x += 1
            points_left += len(pcd.points)
        si_mesh.compute_vertex_normals()
        if color_by_column:
            #print(columns[mat[0,2], mat[0,1], mat[0,0]])
            si_mesh.paint_uniform_color(cmap[i,:])
        else:
            si_mesh.paint_uniform_color(colors['a_si'])
        si_meshes.append(si_mesh)
    print("\t{} able to mesh; {} unable for {} points left out.".format(n, x, points_left))
    print("Created surface meshes.")
    return si_meshes, ag_meshes