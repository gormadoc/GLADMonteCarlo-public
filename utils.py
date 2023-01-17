import math
import numpy as np
import gzip
import time
import json
import copy


def cosd(angle):
    return math.cos(angle*math.pi/180.)


def sind(angle):
    return math.sin(angle*math.pi/180.)


def tand(angle):
    return math.tan(angle*math.pi/180.)

def getLinePoints(src, dest):
    #src = Point(src1.y, src1.x)
    #dest = Point(dest1.y, dest1.x)
    steep = abs(dest.y - src.y) > abs(dest.x - src.x)
    
    if steep:
        # swap slope
        t = src.x
        src.x = src.y
        src.y = t
        t = dest.x
        dest.x = dest.y
        dest.y = t
        
    if src.x > dest.x:
        # swap direction
        t = src
        src = dest
        dest = t
        
    dx = dest.x - src.x
    dy = abs(dest.y - src.y)
    error = 0
    y = src.y
    if(src.y < dest.y):
        ystep = 1
    else:
        ystep = -1
    pointlist = list();
    
    # iterate over all x and grab nearest y for next point
    # steep-swapping allows for finer x selection over near-vertical lines
    for x in range(src.x, dest.x+1):
        if steep:
            pointlist.append(Point(y, x))
        else:
            pointlist.append(Point(x, y))
        error += dy
        if (2*error >= dx):
            y += ystep
            error -= dx
    return pointlist

def getLinePointsP(src1, dest1):
    #src = Point(src1.y, src1.x)
    #dest = Point(dest1.y, dest1.x)
    src = [src1[0], src1[1]]
    dest = [dest1[0], dest1[1]]
    dx = dest1[0] - src1[0]
    dy = abs(dest1[1] - src1[1])
    dxa = abs(dx)
    steep = dy > dxa
    
    if steep:
        # swap slope
        t = src[0]
        src[0] = src[1]
        src[1] = t
        t = dest[0]
        dest[0] = dest[1]
        dest[1] = t
        
    if src[0] > dest[0]:
        # swap direction
        t = src
        src = dest
        dest = t
    
    error = 0
    y = src[1]
    if(src[1] < dest[1]):
        ystep = 1
    else:
        ystep = -1
    pointlist = [None]*int(dy + dxa + 1);
    
    # iterate over all x and grab nearest y for next point
    # steep-swapping allows for finer x selection over near-vertical lines
    i = 0
    for x in range(src[0], dest[0]+1):
        if steep:
            pointlist[i] = (y, x)
        else:
            pointlist[i] = (x, y)
        i += 1
        error += dy
        if (2*error >= dx):
            y += ystep
            error -= dx
    return pointlist


def Bresenham3D(src, dest):
    dx = abs(dest[0] - src[0])
    dy = abs(dest[1] - src[1])
    dz = abs(dest[2] - src[2])
    pointlist = [None]*int(dx+dy+dz+10)
    pointlist[0] = np.array([src[0], src[1], src[2]], dtype=np.float32)
    if (dest[0] > src[0]):
        xs = 1
    else:
        xs = -1
    if (dest[1] > src[1]):
        ys = 1
    else:
        ys = -1
    if (dest[2] > src[2]):
        zs = 1
    else:
        zs = -1
    x1 = src[0]
    y1 = src[1]
    z1 = src[2]
    x2 = dest[0]
    y2 = dest[1]
    z2 = dest[2]
    
    i = 1
    # Driving axis is X-axis"
    if (dx >= dy and dx >= dz):  
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while (x1 != x2):
            x1 += xs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dx
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            try:
                pointlist[i] = np.array([x1, y1, z1], dtype=np.float32)
            except IndexError:
                pointlist.append(np.array([x1, y1, z1], dtype=np.float32))
            i += 1
    # Driving axis is Y-axis"
    elif (dy >= dx and dy >= dz):
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while (y1 != y2):
            y1 += ys
            if (p1 >= 0):
                x1 += xs
                p1 -= 2 * dy
            if (p2 >= 0):
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            try:
                pointlist[i] = np.array([x1, y1, z1], dtype=np.float32)
            except IndexError:
                pointlist.append(np.array([x1, y1, z1], dtype=np.float32))
            i += 1
    # Driving axis is Z-axis"
    else:        
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while (z1 != z2):
            z1 += zs
            if (p1 >= 0):
                y1 += ys
                p1 -= 2 * dz
            if (p2 >= 0):
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            try:
                pointlist[i] = np.array([x1, y1, z1], dtype=np.float32)
            except IndexError:
                pointlist.append(np.array([x1, y1, z1], dtype=np.float32))
            i += 1
    return pointlist

''' 3D Grid functions '''
def saveGrid(grid, deposited, params, system, zipped=True, infix=''):
    filename = ''
    if infix != '':
        infix += '_'
    ps = copy.deepcopy(params)
    for p in ps:
        if isinstance(p['weights'], np.ndarray):
            p['weights'] = p['weights'].tolist()
    dep = sum(deposited)
    time_now = int(round(time.time()))
    turns = 0
    L = 0
    thetas = []
    bent_infix = ''
    for p in ps:
        if 'turns' in p:
            turns += p['turns']
        if p['L'] > L:
            L = p['L']
        #if p['theta'] not in thetas:
        #    bent_infix = '_sc'
    p = params[-1]
    if turns == 0:
        filename += 'STF_{}_{}L{}_Th{}_D{}_N{}_{}'.format(system + bent_infix, infix, L, p['theta'], p['D'], dep, time_now)
    else:
        filename += 'STF_{}_{}L{}_x{}_Th{}_D{}_N{}_{}'.format(system, infix, L, turns, p['theta'], p['D'], dep, time_now)
    json_data = {"Layers": len(ps), "Total points": dep}
    for i in range(1, len(ps)+1):
        json_data[i] = {"Deposited": deposited[i-1], "Parameters": ps[i-1]}
    with open('structures/' + filename + ".json", 'w') as f:
        json.dump(json_data, f)
    if zipped:
        np.savez_compressed('structures/' + filename + ".npz", grid)
        return filename
    else:
        np.save('structures/' + filename + '.npy', grid)
        return filename

def loadSparse(filename):
    if filename[-3:] == 'npz':
        jf = filename[:-3] + 'json'
    elif filename[-3:] == 'npy':
        jf = filename[:-3] + 'json'
    else:
        jf = ''
    params = []
    deposited = []
    try:
        with open(jf, 'r') as f:
            data = json.load(f)
        i = 1
        while "{}".format(i) in data.keys():
            params.append(data["{}".format(i)]['Parameters'])
            deposited.append(data["{}".format(i)]['Deposited'])
            i += 1
    except OSError:
        print("No JSON file paired with data.")
    for p in params:
        p['weights'] = np.array(p['weights'])
    if filename[-3:] == 'npz':
        return np.load(filename, allow_pickle=True)['arr_0'], deposited, params
    else:
        return np.load(filename), deposited, params
    

def loadGridDense(filename):
    if filename[-2:] == 'gz':
        jf = filename[:-2] + 'json'
    elif filename[-3:] == 'npy':
        jf = filename[:-3] + 'json'
    else:
        jf = ''
    params = []
    deposited = []
    try:
        with open(jf, 'r') as f:
            data = json.load(f)
        i = 1
        while "Layer {}".format(i) in data.keys():
            params.append(data["Layer {}".format(i)]['Parameters'])
            deposited.append(data["Layer {}".format(i)]['Deposited'])
            i += 1
    except OSError:
        print("No JSON file paired with data.")
    for p in params:
        p['weights'] = np.array(p['weights'])
    if filename[-2:] == 'gz':
        with gzip.GzipFile(filename, 'rb') as f:
            return np.load(filename, allow_pickle=True), deposited, params
    else:
        with open(filename, 'rb') as f:
            return np.load(filename), deposited, params
        
def matrixFromSparse(inputGrid):
    m = np.amax(inputGrid, axis=0)+1
    grid = np.zeros([m[2], m[1], m[0]]).astype(np.int32)
    for i in range(inputGrid.shape[0]):
        x = inputGrid[i,:]
        grid[x[2], x[1], x[0]] = x[3]
    return grid
        
# takes valued [Z, Y, X] matrix and converts to [x, y, z, value] point list
def sparseFromMatrix(inputGrid):
    l = np.flip(np.array(np.where(inputGrid!=0), dtype=np.int32).T, axis=1)
    grid = np.zeros([l.shape[0], 4], dtype=np.int32)
    grid[:,0:3] = l
    for i in range(grid.shape[0]):
        x = grid[i,:]
        x[3] = inputGrid[x[2], x[1], x[0]]
    return grid

''' Continuous space simulations '''
def loadContinuum(filename, folder = 'structures/continuum'):
    if folder != '':
        folder += '/'
        
    if filename[-3:] == 'npz':
        jf = filename[:-3] + 'json'
    elif filename[-3:] == 'npy':
        jf = filename[:-3] + 'json'
    else:
        jf = ''
    params = []
    deposited = []
    try:
        with open(folder + jf, 'r') as f:
            data = json.load(f)
        i = 1
        while "{}".format(i) in data.keys():
            params.append(data["{}".format(i)]['Parameters'])
            deposited.append(data["{}".format(i)]['Deposited'])
            i += 1
    except OSError:
        print("No JSON file paired with data.")
    for p in params:
        p['weights'] = np.array(p['weights'])
    if filename[-3:] == 'npz':
        return np.load(folder + filename, allow_pickle=True)['arr_0'], deposited, params
    else:
        return np.load(folder + filename), deposited, params
    
def saveContinuum(atoms, params, system, folder='structures/continuum', zipped=True, infix=''):
    filename = ''
    if infix != '':
        infix += '_'
    if folder != '':
        folder += '/'
    
    data = atoms#np.column_stack([positions[:,:3], species.astype(np.float32), positions[:,3]])
    
    ps = copy.deepcopy(params)
    for p in ps:
        if isinstance(p['weights'], np.ndarray):
            p['weights'] = p['weights'].tolist()
    deposited = int(atoms.shape[0])
    time_now = int(round(time.time()))
    turns = 0
    L = 0
    thetas = []
    bent_infix = ''
    for p in ps:
        if 'turns' in p:
            turns += p['turns']
        if p['L'] > L:
            L = p['L']
    p = params[-1]
    if turns == 0:
        filename += 'STF_{}_{}L{}_Th{}_D{}_N{}_{}'.format(system + bent_infix, infix, L, p['theta'], p['D'], deposited, time_now)
    else:
        filename += 'STF_{}_{}L{}_x{}_Th{}_D{}_N{}_{}'.format(system, infix, L, turns, p['theta'], p['D'], deposited, time_now)
    json_data = {"Layers": len(ps), "Total points": deposited}
    for i in range(1, len(ps)+1):
        json_data[i] = {"Deposited": deposited, "Parameters": ps[i-1]}
    with open(folder + filename + ".json", 'w') as f:
        json.dump(json_data, f)
    if zipped:
        np.savez_compressed(folder + filename + ".npz", data)
        return folder + filename
    else:
        np.save(folder + filename + '.npy', data)
        return folder + filename
        
''' 2D grid functions '''
def loadGrid2D(filename, flip=True):
    if filename[-2:] == 'gz':
        jf = 'structures/2D/' + filename[:-2] + 'json'
    elif filename[-3:] == 'npy':
        jf = 'structures/2D/' + filename[:-3] + 'json'
    else:
        jf = ''
    params = []
    deposited = []
    try:
        with open(jf, 'r') as f:
            data = json.load(f)
        i = 1
        while "{}".format(i) in data.keys():
            params.append(data["{}".format(i)]['Parameters'])
            deposited.append(data["{}".format(i)]['Deposited'])
            i += 1
    except OSError:
        print("No JSON file paired with data.")
    for p in params:
        p['weights'] = np.array(p['weights'])
    if filename[-2:] == 'gz':
        with gzip.GzipFile('structures/2D/' + filename, 'rb') as f:
            grid = np.load('structures/2D/' + filename, allow_pickle=True)
            if flip:
                grid = np.flip(grid[:,:], axis=0).astype(np.uint8)
    else:
        with open('structures/2D/' + filename, 'rb') as f:
            grid = np.load('structures/2D/' + filename)
            if flip:
                grid = np.flip(grid[:,:], axis=0).astype(np.uint8)
    return grid, deposited, params

def get_bounding_box(array, val):
    """
    Get low and high points for bounding box of a dense matrix.
        Parameters:
            array (ZxYxX ndarray): Dense 3D matrix
            val: Value to contain within box.
        Returns:
            ll (1x3 ndarray): Low points [x, y, z].
            ur (1x3 ndarray): Low points [x, y, z].
    """
    indices = np.argwhere(array == val)
    ll = (np.min(indices[:,0]), np.min(indices[:,1]), np.min(indices[:,2]))
    ur = (np.max(indices[:,0]), np.max(indices[:,1]), np.max(indices[:,2]))
    return ll, ur