import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt

mesh = trimesh.load('resources/SR_Gripper_Collision_Open.stl')
color = (np.asarray((0.8, 0.5, 0.5)))
mesh.visual.face_colors = np.tile(
    np.reshape(color, [1, 3]), [mesh.faces.shape[0], 1]
)
mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
scene = pyrender.Scene()
scene.add(mesh)
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
s = np.sqrt(2)/2
camera_pose = np.array([
   [0.0, -s,   s,   0.3],
   [1.0,  0.0, 0.0, 0.0],
   [0.0,  s,   s,   0.35],
   [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
scene.add(light, pose=camera_pose)

r = pyrender.OffscreenRenderer(400, 400)
color, depth = r.render(scene)
plt.figure()
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(color)
plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(depth, cmap=plt.cm.gray_r)
plt.show()