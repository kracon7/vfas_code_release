import os
import argparse
import h5py
import yaml
import time
import trimesh
import pyrender
from utils import SceneRenderer, wait_for_user_input
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy

np.set_printoptions(precision=1, suppress=True)

def add_origin_frame(origin=[0,0,0]):
    cx = trimesh.creation.cylinder(
        radius=0.004,
        sections=6,
        segment=[origin, [origin[0]+0.1, origin[1], origin[2]]],
    )
    cx.visual.face_colors = [200,0,0]
    cy = trimesh.creation.cylinder(
        radius=0.004,
        sections=6,
        segment=[origin, [origin[0], origin[1]+0.1, origin[2]]],
    )
    cy.visual.face_colors = [0,200,0]
    cz = trimesh.creation.cylinder(
        radius=0.004,
        sections=6,
        segment=[origin, [origin[0], origin[1], origin[2]+0.1]],
    )
    cz.visual.face_colors = [0,0,200]
    tmp = trimesh.util.concatenate([cx, cy, cz])
    return tmp

def add_grid(origin=[0,0,0], d=0.05):
    meshes = []
    for i in range(10):
        gxy = trimesh.creation.cylinder(
            radius=0.001,
            sections=6,
            segment=[[origin[0]-0.25, origin[1] + i * d-0.25, origin[2]], 
                     [origin[0]+0.25, origin[1] + i * d-0.25, origin[2]]],
        )
        gxy.visual.face_colors = [255,0,0]
        meshes.append(gxy)
        gxy = trimesh.creation.cylinder(
            radius=0.001,
            sections=6,
            segment=[[origin[0] + i * d-0.25, origin[1]-0.25, origin[2]], 
                     [origin[0] + i * d-0.25, origin[1]+0.25, origin[2]]],
        )
        gxy.visual.face_colors = [0,255,0]
        meshes.append(gxy)
    tmp = trimesh.util.concatenate(meshes)
    return tmp

def main(args):
    renderer = SceneRenderer() 

    object_info = h5py.File(args.info, 'r')

    # Content to output for mesh white list
    content = {}

    with open(args.instances, 'r') as f:
        obj_instances = yaml.load(f, Loader=yaml.Loader)
        obj_cats = obj_instances['white_list']

    origin_frame = add_origin_frame([0,0,0])
    renderer.add_object('origin', origin_frame, np.eye(4))
    grid = add_grid()
    renderer.add_object('grid_xy', grid, np.eye(4))

    v = pyrender.Viewer(renderer._scene, run_in_thread=True, use_raymond_lighting=True)

    for cat in obj_cats:
        print("\n\n================  %s  ==================\n"%cat)
        content[cat] = []
        obj_keys = object_info['categories'][cat][()]

        for obj_key in obj_keys:
            # Load object mesh
            mesh_path = os.path.join(os.path.dirname(args.info), 
                                     object_info['meshes'][obj_key]['path'][()].decode())
            mesh = trimesh.load(mesh_path)

            # Remove last object mesh and new one
            v.render_lock.acquire()
            if renderer.has_object('obj_mesh'):
                renderer.remove_object('obj_mesh')
            renderer.add_object('obj_mesh', mesh, pose=np.eye(4))
            v.render_lock.release()

            x = print("Add mesh?\n press 'a' to confirm, press other keys to skip")
            x = wait_for_user_input()
            if x == 'a':
                content[cat].append(obj_key.decode())

    # Close viewer
    v.close_external()
    while v.is_active:
        pass
    
    with open(args.output, 'w') as file:
        documents = yaml.dump(content, file)

    # white_list = []

    # x = ''
    # while x != 's':

    #     cat = input("Enter object category: ")
    #     if cat == 's':
    #         break
    #     if not cat in list(obj_cats):
    #         print("Category %s does not exist")
    #         continue
        
    #     obj_keys = object_info['categories'][cat][()]
    #     print("Visualize category %s with %d meshes"%(cat, len(obj_keys)))

    #     mesh_list = [draw_frame([0,0,0], [0,0,0,1], scale=0.2)]
    #     for i, obj_key in enumerate(obj_keys):
    #         obj = object_info['meshes'][obj_key]
    #         mesh_path = os.path.join('dataset', obj['path'][()].decode())
    #         w_t_g = np.eye(4)
    #         w_t_g[:3,3] += (i % ncol) * np.array([.4, 0, 0])
    #         w_t_g[:3,3] += (i // ncol) * np.array([0, .4, 0])
    #         obj_mesh = o3d.io.read_triangle_mesh(mesh_path).transform(w_t_g)

    #         mesh_list.append(copy.deepcopy(obj_mesh))

    #     o3d.visualization.draw_geometries(mesh_list)

        

    # object_info.close()

    # content = {'white_list': white_list}, {'black_list':[]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--info", type=str, default='dataset/object_info.hdf5')
    parser.add_argument("--instances", type=str, default='config/object_instances_whitelist_2.yaml')
    parser.add_argument("--output", type=str, default='config/object_meshes_whitelist.yaml')

    args = parser.parse_args()
    main(args)

