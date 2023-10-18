import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any, Callable, Dict, Iterable
import h5py
import random
import numpy as np
import trimesh
import trimesh.transformations as tra
import pyrender
from autolab_core import ColorImage, DepthImage
from tqdm import tqdm
import sys, termios, tty

def wait_for_user_input(prompt: str = None):
    if prompt:
        print(prompt)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


np.seterr(all='raise')

def is_rotation_matrix(r):
    c1 = np.abs((r @ r.T).trace() - 3) < 1e-3
    c2 = np.abs(np.linalg.det(r) - 1) < 1e-3
    return c1 & c2

def rotation_similarity(r1, r2):
    '''
    Similarity between 2 3D rotation matrix.
    Return the normalized angle in axis-angle representation.
    Args:
        r1, r2 -- (numpy.array) shape: n x 3 x 3
    '''
    r2 = np.transpose(r2, (0,2,1))
    dr = np.matmul(r1, r2)
    trace = np.trace(dr, axis1=1, axis2=2)
    try:
        w = np.arccos((trace - 1) / 2)
        s = 1 - np.abs(w) / np.pi
    except:
        # trace computation might have numerical error
        assert np.all(trace[trace>3] - 3 < 1e-4)
        assert np.all(trace[trace<-1] + 1 > -1e-4)
        trace[trace>3] = 3
        trace[trace<-1] = -1
        w = np.arccos((trace - 1) / 2)
        s = 1 - np.abs(w) / np.pi
    return s

def translation_similarity(p1, p2, cutoff=0.05):
    '''
    Similarity between two translation vectors
    Args:
        p1, p2 -- (numpy.array) shape: n x 3
    '''
    dist = np.linalg.norm(p1 - p2, axis=1)
    dist = 1 - (dist / cutoff)
    dist[dist <= 0] = 0
    return dist
     
    
def transform_similarity(ref, r, cutoff=0.05):
    '''
    Highest similarity between each transformation matrix in r
    and all ref matrix
    Args:
        ref -- (numpy.array) shape: m x 4 x 4 
                reference transform to compare with
        r -- (numpy.array) shape: n x 4 x 4
    '''
    m, n = ref.shape[0], r.shape[0]
    ref = np.tile(np.expand_dims(ref, 1), (1,n,1,1)).reshape(-1,4,4)
    r = np.tile(np.expand_dims(r, 0), (m,1,1,1)).reshape(-1,4,4)
    
    # Rotation similarity
    sr = rotation_similarity(ref[:, :3, :3], r[:, :3, :3])
    # Translation similarity
    sp = translation_similarity(ref[:, :3, 3], r[:, :3, 3], cutoff)

    # Heighest similarity
    sr = np.max(sr.reshape(m,n), axis=0)
    sp = np.max(sp.reshape(m,n), axis=0)

    s = np.multiply(sr, sp)
    s[np.isnan(s)] = 1
    return s

def random_transform(n=1, 
                     tx_ranges=[[-1, 1]], 
                     ty_ranges=[[-1, 1]],
                     tz_ranges=[[-1, 1]], 
                     rx_ranges=[[-1, 1]], 
                     ry_ranges=[[-1, 1]],
                     rz_ranges=[[-1, 1]]):
    '''
    Generate random 3D transform matrix
    Args:
        n -- number of samples
        txrange -- (list) list of ranges for the translation sampling
        rxrange -- (list) list of ranges for the rotation sampling
    '''
    res = []
    for _ in range(n):
        tx_range = random.choice(tx_ranges)
        ty_range = random.choice(ty_ranges)
        tz_range = random.choice(tz_ranges)
        rx_range = random.choice(rx_ranges)
        ry_range = random.choice(ry_ranges)
        rz_range = random.choice(rz_ranges)
        sampled_translation = [
            np.random.uniform(tx_range[0], tx_range[1]),
            np.random.uniform(ty_range[0], ty_range[1]),
            np.random.uniform(tz_range[0], tz_range[1])
        ]
        sampled_rotation = [
            np.random.uniform(rx_range[0], rx_range[1]),
            np.random.uniform(ry_range[0], ry_range[1]),
            np.random.uniform(rz_range[0], rz_range[1])
            
        ]
        grasp_transformation = tra.euler_matrix(*sampled_rotation, 
                                                'szyx')
        grasp_transformation[:3, 3] = sampled_translation

        res.append(grasp_transformation)
    res = np.array(res)
    return res

class MeshLoader(object):
    """A tool for loading meshes from a base directory.
    Attributes
    ----------
    basedir : str
        basedir containing mesh files
    """

    def __init__(self, basedir, only_ext=None):
        self.basedir = basedir
        self._map = {}
        for root, _, fns in os.walk(basedir):
            for fn in fns:
                full_fn = os.path.join(root, fn)
                f, ext = os.path.splitext(fn)
                if only_ext is not None and ext != only_ext:
                    continue
                if basedir != root:
                    f = os.path.basename(root) + "~" + f
                if ext[1:] not in trimesh.available_formats():
                    continue
                if f in self._map:
                    continue
                    # raise ValueError('Duplicate file named {}'.format(f))
                self._map[f] = full_fn

    def meshes(self):
        return self._map.keys()

    def get_path(self, name):
        if name in self._map:
            return self._map[name]
        raise ValueError(
            "Could not find mesh with name {} in directory {}".format(
                name, self.basedir
            )
        )

    def load(self, name):
        m = trimesh.load(self.get_path(name))
        m.metadata["name"] = name
        return m


class ProcessKillingExecutor:
    """
    The ProcessKillingExecutor works like an `Executor
    <https://docs.python.org/dev/library/concurrent.futures.html#executor-objects>`_
    in that it uses a bunch of processes to execute calls to a function with
    different arguments asynchronously.

    But other than the `ProcessPoolExecutor
    <https://docs.python.org/dev/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor>`_,
    the ProcessKillingExecutor forks a new Process for each function call that
    terminates after the function returns or if a timeout occurs.

    This means that contrary to the Executors and similar classes provided by
    the Python Standard Library, you can rely on the fact that a process will
    get killed if a timeout occurs and that absolutely no side can occur
    between function calls.

    Note that descendant processes of each process will not be terminated –
    they will simply become orphaned.
    """

    def __init__(self, max_workers: int = None):
        self.processes = max_workers or os.cpu_count()

    def map(
        self,
        func: Callable,
        iterable: Iterable,
        timeout: float = None,
        callback_timeout: Callable = None,
        daemon: bool = True,
    ) -> Iterable:
        """
        :param func: the function to execute
        :param iterable: an iterable of function arguments
        :param timeout: after this time, the process executing the function
                will be killed if it did not finish
        :param callback_timeout: this function will be called, if the task
                times out. It gets the same arguments as the original function
        :param daemon: define the child process as daemon
        """
        executor = ProcessPoolExecutor(max_workers=self.processes)
        params = (
            {
                "func": func,
                "fn_args": p_args,
                "p_kwargs": {},
                "timeout": timeout,
                "callback_timeout": callback_timeout,
                "daemon": daemon,
            }
            for p_args in iterable
        )
        return executor.map(self._submit_unpack_kwargs, params)

    def _submit_unpack_kwargs(self, params):
        """unpack the kwargs and call submit"""

        return self.submit(**params)

    def submit(
        self,
        func: Callable,
        fn_args: Any,
        p_kwargs: Dict,
        timeout: float,
        callback_timeout: Callable[[Any], Any],
        daemon: bool,
    ):
        """
        Submits a callable to be executed with the given arguments.
        Schedules the callable to be executed as func(*args, **kwargs) in a new
         process.
        :param func: the function to execute
        :param fn_args: the arguments to pass to the function. Can be one argument
                or a tuple of multiple args.
        :param p_kwargs: the kwargs to pass to the function
        :param timeout: after this time, the process executing the function
                will be killed if it did not finish
        :param callback_timeout: this function will be called with the same
                arguments, if the task times out.
        :param daemon: run the child process as daemon
        :return: the result of the function, or None if the process failed or
                timed out
        """
        p_args = fn_args if isinstance(fn_args, tuple) else (fn_args,)
        queue = Queue()
        p = Process(
            target=self._process_run,
            args=(
                queue,
                func,
                *p_args,
            ),
            kwargs=p_kwargs,
        )

        if daemon:
            p.daemon = True

        p.start()
        try:
            ret = queue.get(block=True, timeout=timeout)
            if ret is None:
                callback_timeout(*p_args, **p_kwargs)
            return ret
        except Empty:
            callback_timeout(*p_args, **p_kwargs)
        if p.is_alive():
            p.terminate()
            p.join()

    @staticmethod
    def _process_run(
        queue: Queue, func: Callable[[Any], Any] = None, *args, **kwargs
    ):
        """
        Executes the specified function as func(*args, **kwargs).
        The result will be stored in the shared dictionary
        :param func: the function to execute
        :param queue: a Queue
        """
        queue.put(func(*args, **kwargs))


def compute_camera_pose(distance, azimuth, elevation):
    cam_tf = tra.euler_matrix(np.pi / 2, 0, 0).dot(
        tra.euler_matrix(0, np.pi / 2, 0)
    )

    extrinsics = np.eye(4)
    extrinsics[0, 3] += distance
    extrinsics = tra.euler_matrix(0, -elevation, azimuth).dot(extrinsics)

    cam_pose = extrinsics.dot(cam_tf)
    frame_pose = cam_pose.copy()
    frame_pose[:, 1:3] *= -1.0
    return cam_pose, frame_pose


def process_mesh(in_path, out_path, scale, grasps, return_stps=True):
    mesh = trimesh.load(in_path, force="mesh", skip_materials=True)
    cat = (
        ""
        if os.path.basename(os.path.dirname(out_path)) == "meshes"
        else os.path.basename(os.path.dirname(out_path))
    )
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    if not mesh.is_watertight or len(mesh.faces) > 1000:
        obj_path = os.path.splitext(in_path)[0] + ".obj"
        is_obj = os.path.exists(obj_path)
        if not is_obj:
            mesh.export(obj_path)

        simplify_path = "{}_{:d}.obj".format(
            os.path.splitext(out_path)[0],
            np.random.RandomState().randint(2 ** 16),
        )
        manifold_cmd = [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../extern/Manifold/build/manifold",
            ),
            obj_path,
            simplify_path,
        ]
        simplify_cmd = [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../extern/Manifold/build/simplify",
            ),
            "-i",
            simplify_path,
            "-o",
            simplify_path,
            "-m",
            "-f 1000",
        ]
        try:
            subprocess.check_output(manifold_cmd)
        except subprocess.CalledProcessError:
            if not is_obj:
                os.remove(obj_path)
            if os.path.exists(simplify_path):
                os.remove(simplify_path)
            return None
        if not is_obj:
            os.remove(obj_path)
        try:
            subprocess.check_output(simplify_cmd)
        except subprocess.CalledProcessError:
            if os.path.exists(simplify_path):
                os.remove(simplify_path)
            return None

        mesh = trimesh.load(simplify_path)
        os.remove(simplify_path)

    # Create final scaled and transformed mesh
    if not mesh.is_watertight:
        mesh.center_mass = mesh.centroid

    mesh.apply_scale(scale)
    mesh_offset = mesh.center_mass
    mesh.apply_transform(
        trimesh.transformations.translation_matrix(-mesh_offset)
    )
    m_scale = "{}_{}.obj".format(
        os.path.splitext(os.path.basename(out_path))[0], scale
    )
    s_out_path = os.path.join(os.path.dirname(out_path), m_scale)
    mesh.export(s_out_path)

    m_info = {
        "path": os.path.join(cat, m_scale),
        "scale": scale,
        "category": cat,
    }

    # Calculate stable poses and add grasps
    if return_stps:
        try:
            stps, probs = mesh.compute_stable_poses()
            if not probs.any():
                os.remove(s_out_path)
                return None
            m_info.update({"stps": stps, "probs": probs / probs.sum()})
        except Exception:
            os.remove(s_out_path)
            return None

    if grasps is not None:
        grasp_data = h5py.File(grasps, "r")["grasps"]

        positive_grasps = grasp_data["transforms"][:][
            grasp_data["qualities/flex/object_in_gripper"][:] > 0
        ]
        positive_grasps[:, :3, 3] -= mesh_offset

        negative_grasps = grasp_data["transforms"][:][
            grasp_data["qualities/flex/object_in_gripper"][:] == 0
        ]
        negative_grasps[:, :3, 3] -= mesh_offset
        

    # Translate to the grasp pose frame in SceneGrasp convention
    T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0.1049],[0,0,0,1]]).astype('float')
    m_info["positive_grasps"] = np.matmul(positive_grasps, T)
    m_info["negative_grasps"] = np.matmul(negative_grasps, T)
    m_info["obj_movement_t"] = grasp_data['qualities/flex/object_motion_during_closing_linear'][:][
        grasp_data["qualities/flex/object_in_gripper"][:] > 0
    ]
    m_info["obj_movement_r"] = grasp_data['qualities/flex/object_motion_during_closing_angular'][:][
        grasp_data["qualities/flex/object_in_gripper"][:] > 0
    ]

    # Augment grasps with random 3D transform
    grasps = []
    ref_grasps = np.concatenate([m_info["positive_grasps"], 
                           m_info["negative_grasps"]], axis=0)
    n = ref_grasps.shape[0]
    
    hard_negative = []
    valid_neg_counter = 0
    #Generate hard negatives in a batch until we have more than pos+neg combined
    while valid_neg_counter < 1000:
        rel_T = random_transform(n, 
                                tx_ranges=[[-0.04, 0.04]], 
                                ty_ranges=[[-0.04, 0.04]],
                                tz_ranges=[[0.05, 0.1], [-0.1, -0.03]], 
                                rx_ranges=[[-0.1, 0.1]], 
                                ry_ranges=[[-0.1, 0.1]],
                                rz_ranges=[[-0.1, 0.1]])
        aug_grasps = np.matmul(ref_grasps, rel_T)
        s = transform_similarity(ref_grasps, aug_grasps)
        valid = s < 0.75
        valid_neg_counter += valid.sum()
        hard_negative.append(aug_grasps[valid])
        
    m_info["hard_negative_grasps"] = np.concatenate(hard_negative, axis=0)

    print("Number of positive: %d, negative: %d, hard negative: %d"%(
            m_info["positive_grasps"].shape[0],
            m_info["negative_grasps"].shape[0],
            m_info["hard_negative_grasps"].shape[0]))
        
    return os.path.splitext(m_scale)[0], m_info




class SceneRenderer:
    def __init__(self):

        self._scene = pyrender.Scene()
        self._node_dict = {}
        self._camera_intr = None
        self._camera_node = None
        self._light_node = None
        self._renderer = None

    def create_camera(self, intr, znear, zfar):
        cam = pyrender.IntrinsicsCamera(
            intr.fx, intr.fy, intr.cx, intr.cy, znear, zfar
        )
        self._camera_intr = intr
        self._camera_node = pyrender.Node(camera=cam, matrix=np.eye(4))
        self._scene.add_node(self._camera_node)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        self._light_node = pyrender.Node(light=light, matrix=np.eye(4))
        self._scene.add_node(self._light_node)
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=intr.width,
            viewport_height=intr.height,
            point_size=5.0,
        )

    @property
    def camera_pose(self):
        if self._camera_node is None:
            return None
        return self._camera_node.matrix

    @camera_pose.setter
    def camera_pose(self, cam_pose):
        if self._camera_node is None:
            raise ValueError("No camera in scene!")
        self._scene.set_pose(self._camera_node, cam_pose)
        self._scene.set_pose(self._light_node, cam_pose)

    def render_rgbd(self, depth_only=False):

        if depth_only:
            depth = self._renderer.render(
                self._scene, pyrender.RenderFlags.DEPTH_ONLY
            )
            color = None
            depth = DepthImage(depth, frame="camera")
        else:
            color, depth = self._renderer.render(self._scene)
            color = ColorImage(color, frame="camera")
            depth = DepthImage(depth, frame="camera")

        return color, depth

    def render_segmentation(self, full_depth=None):
        if full_depth is None:
            _, full_depth = self.render_rgbd(depth_only=True)

        self.hide_objects()
        output = np.zeros(full_depth.data.shape, dtype=np.uint8)
        for i, obj_name in enumerate(self._node_dict):
            self._node_dict[obj_name].mesh.is_visible = True
            _, depth = self.render_rgbd(depth_only=True)
            mask = np.logical_and(
                (np.abs(depth.data - full_depth.data) < 1e-6),
                np.abs(full_depth.data) > 0,
            )
            if np.any(output[mask] != 0):
                raise ValueError("wrong label")
            output[mask] = i + 1
            self._node_dict[obj_name].mesh.is_visible = False
        self.show_objects()

        return output, ["BACKGROUND"] + list(self._node_dict.keys())

    def render_points(self):
        _, depth = self.render_rgbd(depth_only=True)
        point_norm_cloud = depth.point_normal_cloud(self._camera_intr)

        pts = point_norm_cloud.points.data.T.reshape(
            depth.height, depth.width, 3
        )
        norms = point_norm_cloud.normals.data.T.reshape(
            depth.height, depth.width, 3
        )
        cp = self.get_camera_pose()
        cp[:, 1:3] *= -1

        pt_mask = np.logical_and(
            np.linalg.norm(pts, axis=-1) != 0.0,
            np.linalg.norm(norms, axis=-1) != 0.0,
        )
        pts = tra.transform_points(pts[pt_mask], cp)
        return pts.astype(np.float32)

    def add_object(self, name, mesh, pose=None):
        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        node = pyrender.Node(
            name=name,
            mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False),
            matrix=pose,
        )
        self._node_dict[name] = node
        self._scene.add_node(node)

    def add_points(self, points, name, pose=None, color=None, radius=0.005):
        points = np.asanyarray(points)
        if points.ndim == 1:
            points = np.array([points])

        if pose is None:
            pose = np.eye(4)
        else:
            pose = pose.matrix

        color = (
            np.asanyarray(color, dtype=np.float) if color is not None else None
        )

        # If color specified per point, use sprites
        if color is not None and color.ndim > 1:
            self._renderer.point_size = 1000 * radius
            m = pyrender.Mesh.from_points(points, colors=color)
        # otherwise, we can make pretty spheres
        else:
            mesh = trimesh.creation.uv_sphere(radius, [20, 20])
            if color is not None:
                mesh.visual.vertex_colors = color
            poses = None
            poses = np.tile(np.eye(4), (len(points), 1)).reshape(
                len(points), 4, 4
            )
            poses[:, :3, 3::4] = points[:, :, None]
            m = pyrender.Mesh.from_trimesh(mesh, poses=poses)

        node = pyrender.Node(mesh=m, name=name, matrix=pose)
        self._node_dict[name] = node
        self._scene.add_node(node)

    def set_object_pose(self, name, pose):
        self._scene.set_pose(self._node_dict[name], pose)

    def has_object(self, name):
        return name in self._node_dict

    def remove_object(self, name):
        self._scene.remove_node(self._node_dict[name])
        del self._node_dict[name]

    def show_objects(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.is_visible = True

    def toggle_wireframe(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.primitives[0].material.wireframe ^= True

    def hide_objects(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.is_visible = False

    def reset(self):
        for name in self._node_dict:
            self._scene.remove_node(self._node_dict[name])
        self._node_dict = {}


