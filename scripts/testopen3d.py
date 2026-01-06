import open3d as o3d
import numpy as np

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

pc = o3d.geometry.PointCloud()
P = np.array([[0,0,0],[0.2,0,0],[0,0.2,0],[0,0,0.2]], dtype=np.float64)
pc.points = o3d.utility.Vector3dVector(P)
pc.paint_uniform_color([1, 0, 1])  # bright magenta

vis = o3d.visualization.Visualizer()
vis.create_window("O3D test", width=1280, height=720)

vis.add_geometry(mesh)
vis.add_geometry(pc)

opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0], dtype=np.float64)  # force black bg
opt.point_size = 20.0                                          # force big points

vis.reset_view_point(True)  # fit camera

vis.run()
vis.destroy_window()
