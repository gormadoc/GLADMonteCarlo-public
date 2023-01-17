from PyQt5 import QtWidgets, QtGui,QtCore
import open3d as o3d
import win32gui
import sys
import pathlib

from utils import *

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)
        self.setCentralWidget(widget)
        
        self.silver = np.array([0,1,0])
        self.a_si = np.array([1,0,0])

        self.directory = pathlib.Path(__file__).parent.resolve()
        self.filename = "C:/Users/Shawn/GLAD/structures/STF_Si_Ag_L768_Th85.5_D30_N18874368_1659243875.npz"
        self.load_point_clouds(self.filename)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.SiCloud)
        self.vis.add_geometry(self.AgCloud)
        
        self.btn = QtWidgets.QPushButton("Load simulation")
        self.btn.clicked.connect(self.loadSimulation)

        hwnd = win32gui.FindWindowEx(0, 0, None, "Open3D")
        self.window = QtGui.QWindow.fromWinId(hwnd)    
        self.windowcontainer = self.createWindowContainer(self.window, widget)
        layout.addWidget(self.windowcontainer, 0, 0)
        layout.addWidget(self.btn)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_vis)
        timer.start(1)
        
    def load_point_clouds(self, filename):
        grid, deposited, params = loadSparse(filename)
        m = np.max(grid, axis=0)
        grid_full = matrixFromSparse(grid)
        self.si = np.where(grid_full == 1, 1, 0)
        self.ag = np.where(grid_full == 2, 1, 0)
        self.si = np.flip(np.argwhere(self.si == 1), axis=1)
        self.ag = np.flip(np.argwhere(self.ag == 1), axis=1)
        
        self.SiCloud = o3d.geometry.PointCloud()
        self.SiCloud.points = o3d.utility.Vector3dVector(self.si)
        self.SiCloud.paint_uniform_color(self.a_si)
        
        self.AgCloud = o3d.geometry.PointCloud()
        self.AgCloud.points = o3d.utility.Vector3dVector(self.ag)
        self.AgCloud.paint_uniform_color(self.silver)
        
        m = np.max(self.si, axis=0)
        pts = np.asarray(self.SiCloud.points)
        colors = np.asarray(self.SiCloud.colors)
        colors[:,:] = self.a_si
        colors = np.column_stack([0.70*colors[:,0] * pts[:,2]/m[2]+.15, 0.85*colors[:,1] * pts[:,2]/m[2]+.15, 0.85*colors[:,2] * pts[:,2]/m[2]+.15])
        self.SiCloud.colors = o3d.utility.Vector3dVector(colors)
        
        m = np.max(self.ag, axis=0)
        n = np.min(self.ag, axis=0)
        pts = np.asarray(self.AgCloud.points)
        colors = np.asarray(self.AgCloud.colors)
        colors[:,:] = self.silver
        colors = np.column_stack([0.9*colors[:,0] * (pts[:,2]-n[2])/(m[2]-n[2])+.1, 0.85*colors[:,1] * (pts[:,2]-n[2])/(m[2]-n[2])+.15, 0.8*colors[:,2] * (pts[:,2]-n[2])/(m[2]-n[2])+.2])
        self.AgCloud.colors = o3d.utility.Vector3dVector(colors)
        
    def load_new_point_clouds(self, filename):
        self.vis.remove_geometry(self.SiCloud)
        self.vis.remove_geometry(self.AgCloud)
        print("Loading new point clouds")
        self.load_point_clouds(filename)
        self.vis.add_geometry(self.SiCloud)
        self.vis.add_geometry(self.AgCloud)
        
        #self.vis.update_geometry()
        
    def loadSimulation(self):
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Choose simulation', 'structures\\', "Simulation (*.npz)")[0]
        self.load_new_point_clouds(self.filename)

    def update_vis(self):
        #self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    form = MainWindow()
    form.setWindowTitle('o3d Embed')
    form.setGeometry(100, 100, 600, 500)
    form.show()
    sys.exit(app.exec_())