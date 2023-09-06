import os
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import pyrender
import trimesh

from utils import batch_rodrigues, perspective_projection


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces

    def render(self, vertices, vertex_color, alpha, beta, gamma):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.2,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        rot = trimesh.transformations.rotation_matrix(np.radians(alpha), [1, 0, 0]) @\
                trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        vertices = vertices @ rot[:3,:3].transpose()
        mesh = trimesh.Trimesh(vertices, self.faces, vertex_colors=vertex_color)

        mesh = pyrender.Mesh.from_trimesh(mesh)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([[0, 0, 0]])
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) 

        return color, rend_depth
