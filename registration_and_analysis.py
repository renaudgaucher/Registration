from typing import Union, Any

import trimesh.util

from functions import transform_affine, create_mesh, load, rotation
import numpy as np
from cylinder_detection import cylinder_detection
import open3d as o3d
from pathlib import Path
import polyscope as ps
import pandas as pd
import time

DATA_PATH = Path("/home/renaud/Desktop/Data")
PATH_MRI = DATA_PATH / "MRI masks for atlas"
PATH_US = DATA_PATH / "atlas_ICP_manual"  # US segmentations
OUTPUT_PATH = DATA_PATH / "outputs" / "analysis"

patient_index = [i for i in range(16)]
left_index = [2 * i for i in range(16)]
right_index = [2 * i + 1 for i in range(16)]


def main():
    cylinders_left = []
    cylinders_right = []
    cylinders_both = []

    for i in patient_index : #[11, 8, 7, 12, 14, 6]:
        left_path = (PATH_US / f"{left_index[i]}-labels.nii")
        right_path = (PATH_US / f"{right_index[i]}-labels.nii")

        left_final_affine, right_final_affine, cyl_left, cyl_right = register(left_path, right_path, OUTPUT_PATH,
                                                                              display_registration=False,
                                                                              recompute_normals=True, verbose=1)

        cylinder = detect_trachea_when_registered(left_path, right_path, OUTPUT_PATH / "reference",
                                                  display_registration=False,
                                                  recompute_normals=True, verbose=1)

        # Loading nifti images, modifying affine and saving it into an outputs folder
        left_path_image = str(PATH_US / f"{left_index[i]}.nii")
        right_path_image = str(PATH_US / f"{right_index[i]}.nii")

        nifti_left_image, nifti_right_image = load([left_path_image, right_path_image])
        nifti_left, nifti_right = load([left_path, right_path])

        nifti_left._affine = left_final_affine
        nifti_left_image._affine = left_final_affine
        nifti_right_image._affine = right_final_affine
        nifti_right._affine = right_final_affine

        for nifti in [nifti_left, nifti_left_image, nifti_right, nifti_right_image]:
            nifti.update_header()

        nifti_left_image.to_filename(str(DATA_PATH / "outputs" / f"{left_index[i]}.nii"))
        nifti_left.to_filename(str(DATA_PATH / "outputs" / f"{left_index[i]}-labels.nii"))
        nifti_right_image.to_filename(str(DATA_PATH / "outputs" / f"{right_index[i]}.nii"))
        nifti_right.to_filename(str(DATA_PATH / "outputs" / f"{right_index[i]}-labels.nii"))

        cylinders_left.append(cyl_left)
        cylinders_right.append(cyl_right)
        cylinders_both.append(cylinder)

    # center_r, direction_r, r_r, length_r
    df = pd.DataFrame(
        data=
        {
            "radius_both": [cylinders_both[k][2] for k in patient_index],
            "radius_right": [cylinders_right[k][2] for k in patient_index],
            "radius_left": [cylinders_left[k][2] for k in patient_index],
            "centers_both": [cylinders_both[k][0] for k in patient_index],
            "centers_right": [cylinders_right[k][0] for k in patient_index],
            "centers_left": [cylinders_left[k][0] for k in patient_index],
            "direction_both": [cylinders_both[k][1] for k in patient_index],
            "direction_right": [cylinders_right[k][1] for k in patient_index],
            "direction_left": [cylinders_left[k][1] for k in patient_index]
        }
    )
    df.to_excel(str(OUTPUT_PATH / "cylinders_comparison.xlsx"))


class CallBack:

    def __init__(self, thyroid_number, output_path):
        self._number = thyroid_number
        self._output_path = output_path
        self._iter_number = -1
        self._direction_number = -1

        self._likelihoods = []  # float
        self._circles_radius = []  # float
        self._circles_center = []  # (2,) ndarray
        self._directions_id = []  # int
        self._iteration_id = []  # int
        self._directions = []  # (2,) ndarray

        self.data = None

    def add_entry(self, circle_center, radius, likelihood, direction):
        self._circles_center.append(circle_center)
        self._circles_radius.append(radius)
        self._likelihoods.append(likelihood)
        self._directions_id.append(self._direction_number)
        self._directions.append(direction)
        self._iteration_id.append(self._iter_number)

    def new_direction(self):
        self._direction_number += 1

    def new_iter(self):
        self._direction_number = -1
        self._iter_number += 1

    def to_dataframe(self):
        self.data = pd.DataFrame(
            data=
            {
                "likelihood": self._likelihoods,
                "radius": self._circles_radius,
                "centers": self._circles_center,
                "direction": self._directions,
                "id_direction": self._directions_id,
                "id_iteration": self._iteration_id
            }
        )

        del self._likelihoods, self._circles_center, self._circles_radius, self._directions, self._directions_id
        del self._iteration_id

    def save(self):
        if self.data is None:
            self.to_dataframe()

        self.data.to_excel(self._output_path + f"/{self._number}.xlsx")


def register(path_us_left, path_us_right, output_path, display_registration=True, recompute_normals=True, verbose=0):
    """
    :param recompute_normals: bool, if normals of the thyroid should be recomputed or not
    :param output_path:
    :param path_us_left:
    :param path_us_right:
    :param display_registration:
    :return: (4,4) ndarray, (4,4) ndarray - left and right affine transformation from voxel grid to registered us
    """
    date = time.time()

    # loading data from nifti files and meshing all
    path_left = str(path_us_left)
    path_right = str(path_us_right)
    nifti_left, nifti_right = load([path_left, path_right])

    data_left, data_right = nifti_left.get_fdata()[:, :, :], nifti_right.get_fdata()[:, :, :]
    thyroid_left, thyroid_right = data_left == 1., data_right == 1.

    meshes = create_mesh([thyroid_left, thyroid_right])

    pc_thyroid_left = meshes[0][0]
    faces_thyroid_left = meshes[0][1]
    pc_thyroid_right = meshes[1][0]
    faces_thyroid_right = meshes[1][1]

    del meshes[0][:2], meshes[1][:2]

    pc_thyroid_left = transform_affine(nifti_left.affine, pc_thyroid_left)
    pc_thyroid_right = transform_affine(nifti_right.affine, pc_thyroid_right)

    # Cylinder detection

    callback = CallBack(path_us_left.parts[-1][:-11], str(output_path))

    center_l, direction_l, r_l, length_l, loc_map_l = cylinder_detection(
        point_cloud_3d=pc_thyroid_left, faces=faces_thyroid_left,
        recompute_normals=recompute_normals, verbose=verbose,
        **{'callback': callback}
    )
    callback.save()
    callback = CallBack(path_us_right.parts[-1][:-11], str(output_path))

    center_r, direction_r, r_r, length_r, loc_map_r = cylinder_detection(
        point_cloud_3d=pc_thyroid_right, faces=faces_thyroid_right,
        recompute_normals=recompute_normals, verbose=verbose,
        **{'callback': callback}
    )
    callback.save()

    cylinder_right = center_r, direction_r, r_r, length_r
    cylinder_left = center_l, direction_l, r_l, length_l

    # Generation of cylinder mesh
    direction_target = (direction_r + direction_l) / 2
    direction_target = direction_target / np.linalg.norm(direction_target)

    cylinder_mesh_left = o3d.geometry.TriangleMesh.create_cylinder(radius=r_l,
                                                                   height=length_l)
    R_l = rotation(direction_l)
    cyl_left_verts, cyl_left_faces = np.array(cylinder_mesh_left.vertices), np.array(
        cylinder_mesh_left.triangles)
    cyl_left_verts = cyl_left_verts @ R_l.T + center_l

    cylinder_mesh_right = o3d.geometry.TriangleMesh.create_cylinder(radius=r_r,
                                                                    height=length_r)
    R_r = rotation(direction_r)
    cyl_right_verts, cyl_right_faces = np.array(cylinder_mesh_right.vertices), np.array(cylinder_mesh_right.triangles)
    cyl_right_verts = cyl_right_verts @ R_r.T + center_r

    # Fitting left and right thyroids together using the cylinders axis

    affine_l = np.eye(4)
    affine_l[:3, :3] = rotation(direction_l, direction_target).T
    affine_l[:3, 3] = rotation(direction_l, direction_target).T @ (- center_l) + (center_l + center_r) / 2

    affine_r = np.eye(4)
    affine_r[:3, :3] = rotation(direction_r, direction_target).T
    affine_r[:3, 3] = rotation(direction_r, direction_target).T @ (- center_r) + (center_l + center_r) / 2

    pc_thyroid_left = transform_affine(affine_l, pc_thyroid_left)
    cyl_left_verts = transform_affine(affine_l, cyl_left_verts)
    pc_thyroid_right = transform_affine(affine_r, pc_thyroid_right)
    cyl_right_verts = transform_affine(affine_r, cyl_right_verts)

    if verbose:
        print(f"Registration done in {time.time() - date:0.3f}")

    if display_registration:
        ps.init()
        ps.register_surface_mesh(
            "trachea left", cyl_left_verts, cyl_left_faces)
        ps_mesh_l = ps.register_surface_mesh(
            "thyroid left", pc_thyroid_left, faces_thyroid_left)
        ps_mesh_l.add_scalar_quantity("localization map l", loc_map_l)
        ps.register_surface_mesh(
            "trachea right", cyl_right_verts, cyl_right_faces)
        ps_mesh_r = ps.register_surface_mesh(
            "thyroid right", pc_thyroid_right, faces_thyroid_right)
        ps_mesh_r.add_scalar_quantity("localization map r", loc_map_r)
        # ps.show()

    left_final_affine = affine_l @ nifti_left.affine
    right_final_affine = affine_r @ nifti_right.affine

    return left_final_affine, right_final_affine, cylinder_left, cylinder_right


def detect_trachea_when_registered(path_us_left, path_us_right, output_path, display_registration=True,
                                   recompute_normals=True, save_data=True,
                                   verbose=0):
    """
        :param path_us_left:
        :param path_us_right:
        :param display_registration:
        :return: (4,4) ndarray, (4,4) ndarray - left and right affine transformation from voxel grid to registered us
        """
    date = time.time()

    # loading data from nifti files and meshing all
    path_left = str(path_us_left)
    path_right = str(path_us_right)
    nifti_left, nifti_right = load([path_left, path_right])

    data_left, data_right = nifti_left.get_fdata()[:, :, :], nifti_right.get_fdata()[:, :, :]
    thyroid_left, thyroid_right = data_left == 1., data_right == 1.

    meshes = create_mesh([thyroid_left, thyroid_right])

    pc_thyroid_left = meshes[0][0]
    faces_thyroid_left = meshes[0][1]
    pc_thyroid_right = meshes[1][0]
    faces_thyroid_right = meshes[1][1]

    del meshes[0][:2], meshes[1][:2]

    pc_thyroid_left = transform_affine(nifti_left.affine, pc_thyroid_left)
    pc_thyroid_right = transform_affine(nifti_right.affine, pc_thyroid_right)

    # Cylinder detection
    mesh_whole_thyroid = trimesh.util.concatenate(trimesh.Trimesh(pc_thyroid_right, faces_thyroid_right),
                                                  trimesh.Trimesh(pc_thyroid_left, faces_thyroid_left))
    if save_data:
        callback = CallBack(int(path_us_left.parts[-1][:-11]) // 2, str(output_path))
        params = {'callback': callback}
    else:
        params = {}

    center, direction, r, length, loc_map = cylinder_detection(
        point_cloud_3d=mesh_whole_thyroid.vertices, faces=mesh_whole_thyroid.faces,
        recompute_normals=recompute_normals, verbose=verbose,
        **params
    )
    if save_data:
        callback.save()

    cylinder = center, direction, r, length

    # Generation of cylinder mesh

    cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=r,
                                                              height=length)
    R = rotation(direction)
    cyl_verts, cyl_faces = np.array(cylinder_mesh.vertices), np.array(
        cylinder_mesh.triangles)
    cyl_verts = cyl_verts @ R.T + center

    if verbose:
        print(f"Registration done in {time.time() - date:0.3f}")

    if display_registration:
        ps.init()
        ps.register_surface_mesh(
            "trachea", cyl_verts, cyl_faces)
        ps_mesh_l = ps.register_surface_mesh(
            "thyroid", mesh_whole_thyroid.vertices, mesh_whole_thyroid.faces)
        ps_mesh_l.add_scalar_quantity("localization map", loc_map)
        ps.show()

    return cylinder

main()
