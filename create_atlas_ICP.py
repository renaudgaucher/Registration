"""
Creation of atlas registration with an ICP method to fit the US segmented lobe to each MRI segmented lobe

Saving the corresponding transformation into the affine metadata of the .nii files,
for both segmented images and original images.
"""

from functions import *
from pathlib import Path
import numpy.random as rd

data_path = Path("/home/renaud/Desktop/Data")
path_MRI = data_path / "MRI masks for atlas"
path_US = data_path / "US segmentations"
output_path = data_path / "atlas_ICP"

patient_index = [i for i in range(16)]
left_index = [2 * i for i in range(16)]
right_index = [2 * i + 1 for i in range(16)]

for i in patient_index:
    if (i != 11):
        continue  # The 8th patient (Keppler), wasn't properly MRIsed, half of the thyroid is missing

    # Loading images
    left_path = str(path_US / f"{left_index[i]}-labels.nii")
    right_path = str(path_US / f"{right_index[i]}-labels.nii")
    mri_path = str(path_MRI / f"{i}-labels.nii")

    nifti_left, nifti_right, nifti_mri = load([left_path, right_path, mri_path])

    # Extracting only the thyroid label (and not veins/arteria)
    voxel_left_us, voxel_right_us, voxel_mri = nifti_left.get_fdata() == 1., \
                                               nifti_right.get_fdata() == 1., \
                                               nifti_mri.get_fdata() == 1.

    # Detection of both MRI lobes using a KMeans clustering algorithm
    pc_MRI = extract_point_cloud(voxel_mri)

    voxel_left_mri, voxel_right_mri = extract_lobes_voxel(voxel_mri, pc_MRI)

    if i == 11:  # 11th patient MRI data is flipped for an unknown reason
        voxel_left_mri = np.flip(voxel_left_mri, axis=2)
        voxel_right_mri = np.flip(voxel_right_mri, axis=2)

    # Extraction of the surface of the label maps
    meshes = create_mesh([voxel_left_mri, voxel_right_mri, voxel_left_us, voxel_right_us])

    # 190° rotation around X axis to globally align MRI and US coordinates
    rotation = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    mesh_left_mri, mesh_right_mri, mesh_left_us, mesh_right_us = meshes

    mesh_left_mri[0] = transform_affine(nifti_mri.affine, mesh_left_mri[0])
    mesh_right_mri[0] = transform_affine(nifti_mri.affine, mesh_right_mri[0])
    mesh_left_us[0] = transform_affine(rotation @ nifti_left.affine, mesh_left_us[0])
    mesh_right_us[0] = transform_affine(rotation @ nifti_right.affine, mesh_right_us[0])

    # matching the center of mass of US lobes and MRI ones
    translation_left = mesh_left_mri[0].mean(axis=0) - mesh_left_us[0].mean(axis=0)
    translation_right = mesh_right_mri[0].mean(axis=0) - mesh_right_us[0].mean(axis=0)

    affine_translation_left = np.eye(4)
    affine_translation_right = np.eye(4)
    affine_translation_left[:3, 3] = translation_left
    affine_translation_right[:3, 3] = translation_right

    mesh_left_us[0] = transform_affine(affine_translation_left, mesh_left_us[0])
    mesh_right_us[0] = transform_affine(affine_translation_right, mesh_right_us[0])

    # Calculation of ICP registration
    mesh_left_us[0], icp_left_affine = icp_registration(mesh_left_us[0], mesh_left_mri[0])
    mesh_right_us[0], icp_right_affine = icp_registration(mesh_right_us[0], mesh_right_mri[0])


    nb_points_MRI = max(mesh_right_mri[0].shape[0], mesh_left_mri[0].shape[0])
    ps.init()
    ps.register_point_cloud('MRI left', mesh_left_mri[0])
    ps.register_point_cloud('MRI right', mesh_right_mri[0])
    ps.register_point_cloud('US left', mesh_left_us[0][rd.randint(0,mesh_left_us[0].shape[0],nb_points_MRI),:])
    ps.register_point_cloud('US right', mesh_right_us[0][rd.randint(0,mesh_right_us[0].shape[0],nb_points_MRI),:])
    ps.show()


    # Calculus of the total affine transformation
    left_final_affine = icp_left_affine @ affine_translation_left @ rotation @ nifti_left.affine
    right_final_affine = icp_right_affine @ affine_translation_right @ rotation @ nifti_right.affine

    # Loading nifti images, modifying affine and saving it into an outputs folder
    left_path_image = str(path_US / f"{left_index[i]}.nii")
    right_path_image = str(path_US / f"{right_index[i]}.nii")
    nifti_left_image, nifti_right_image = load([left_path_image, right_path_image])

    nifti_left._affine = left_final_affine
    nifti_left_image._affine = left_final_affine
    nifti_right_image._affine = right_final_affine
    nifti_right._affine = right_final_affine

    for nifti in [nifti_left, nifti_left_image, nifti_right, nifti_right_image]:
        nifti.update_header()

    nifti_left_image.to_filename(str(output_path / f"{left_index[i]}.nii"))
    nifti_left.to_filename(str(output_path / f"{left_index[i]}-labels.nii"))
    nifti_right_image.to_filename(str(output_path / f"{right_index[i]}.nii"))
    nifti_right.to_filename(str(output_path / f"{right_index[i]}-labels.nii"))
