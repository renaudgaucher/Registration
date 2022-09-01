from functions import transform_affine, create_mesh, load, rotation
import numpy as np
from cylinder_detection import cylinder_detection
import open3d as o3d
from pathlib import Path
import polyscope as ps

DATA_PATH = Path("/home/renaud/Desktop/Data")
PATH_MRI = DATA_PATH / "MRI masks for atlas"
PATH_US = DATA_PATH / "US segmentations"
OUTPUT_PATH = DATA_PATH / "outputs" / "analysis"


def main():
    i = 0
    for path in PATH_US.glob("*-labels.nii"):
        if int(path.parts[-1][:-11]) not in [16]:
            continue

        print(f"Working on {path}")
        nifti = load([str(path)])[0]
        if len(nifti.get_fdata().shape)==3:
            data = nifti.get_fdata()[:, :, :]
        else:
            data = nifti.get_fdata()[:, :, 0, :]
        thyroid = data == 1.

        meshes = create_mesh([thyroid])

        del thyroid
        del meshes[0][3]
        del data

        pc_thyroid = meshes[0][0]
        faces_thyroid = meshes[0][1]
        pc_thyroid = transform_affine(nifti.affine, pc_thyroid)

        center, direction, r, length, loc_map = cylinder_detection(pc_thyroid, faces=faces_thyroid,
                                              recompute_normals=True, verbose=2)
        cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=r,
                                                                  height=length)

        rot = rotation(direction)
        verts, faces = np.array(cylinder_mesh.vertices), np.array(cylinder_mesh.triangles)
        verts_cylinder = verts @ rot.T + center

        # ps.init()
        # ps_mesh_l = ps.register_surface_mesh(
        #     "thyroid", pc_thyroid, faces_thyroid)
        # ps_mesh_l.add_scalar_quantity("localization map", loc_map)
        ps.register_surface_mesh(
            "cylinder left", verts_cylinder, faces)
        ps.show()


if __name__ == '__main__':
    main()
