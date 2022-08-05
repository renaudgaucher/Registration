import nibabel as nib
from skimage.measure import marching_cubes
from sklearn.cluster import KMeans
import polyscope as ps
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

import open3d as o3d
from pycpd import RigidRegistration, DeformableRegistration, AffineRegistration
import matplotlib.pyplot as plt


def transform_affine(affine, point_cloud):
    """
    tranform a point cloud using a (4,4) affine transform
    :param affine: (m,3) ndarray
    :param point_cloud: (4,4) ndarray
    :return: (m,3) ndarray
    """

    temp = np.ones((point_cloud.shape[0], 4))
    temp[:, :-1] = point_cloud
    return affine.dot(temp.T).T[:, :-1]


def load(paths):
    """
    load all images which path is in the list paths

    :param paths : string list of path
    """
    print(f"loading data ... {paths}")
    nifti_images = []
    for path in paths:
        nifti_images.append(nib.load(path))
    return nifti_images


def create_mesh(voxel_grid_list):
    """
    create a mesh (verts, faces, normals, values) using the marching cube algorithm from a list of voxel grid
    """
    shape_list = []
    for pc in voxel_grid_list:
        print("creating mesh...")
        shape_list.append(list(marching_cubes(pc)))
    return shape_list


def load_and_mesh(paths, label=1., apply_affine=True):
    """
    load, mesh and transform points coordinates according to metadata all in once
    :param paths: list of path
    :return: list of mesh (verts, faces, normals, values) transformed accordingly to affine information in metadata
    """
    imgs = load(paths)

    imgs_data = []
    for image in imgs:
        if len(image.get_fdata().shape) == 3:
            imgs_data.append(image.get_fdata()[:, :, :])
        elif len(image.get_fdata().shape) == 4:
            imgs_data.append(image.get_fdata()[:, :, 0, :])
        imgs_data[-1] = imgs_data[-1] * (imgs_data[-1] == label)
    shape_list = create_mesh(imgs_data)

    for i in range(len(shape_list)):
        if apply_affine:
            shape_list[i][0] = transform_affine(imgs[i].affine, shape_list[i][0])
    return shape_list


def extract_point_cloud(voxel_grid, affine=None, label=1.):
    """
    Extract a point cloud from a binary voxel grid, without meshing, and return a point cloud
    :param voxel_grid: L,H,C,D or L,H,D voxel grid (channel considered = 0)
    :param affine: affine transform np.array(4,4)
    :return: point cloud np.array(3, nb_point)
    """
    print("extracting point cloud")
    if (len(voxel_grid.shape) == 4):
        voxel = voxel_grid[:, :, 0, :]
    else:
        voxel = voxel_grid[:, :, :]

    x = np.linspace(0, voxel.shape[0])
    y = np.linspace(0, voxel.shape[1])
    z = np.linspace(0, voxel.shape[2])

    pc = np.argwhere(voxel == label)
    if affine is not None:
        pc = transform_affine(affine, pc)
    return pc


def extract_lobes_pc(point_cloud):
    """
    Detect lobes from a complete thyroid using a KMeans algorithm implemented in scikit-learn
    :param point_cloud:
    :return: point_cloud, point_cloud
    """
    print("detecting lobes")
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(point_cloud)
    return point_cloud[kmeans.labels_.astype(bool), :], \
           point_cloud[(1 - kmeans.labels_).astype(bool), :]


def extract_lobes_voxel(voxel_grid, point_cloud=None):
    """
    Detect lobes from a complete thyroid using a KMeans algorithm implemented in scikit-learn
    #assumed that on axis 0, left point cloud position < right point cloud position
    :param voxel_grid:
    :return: voxel_grid_left, voxel_grid_right (same shape as input)
    """

    if point_cloud is None:
        pc = extract_point_cloud(voxel_grid)
    else:
        pc = point_cloud

    pc_left, pc_right = extract_lobes_pc(pc)

    if (pc_left.mean(axis=0)[0] < pc_right.mean(axis=0)[0]):
        pc_left, pc_right = pc_right, pc_left

    data_left, data_right = np.zeros(voxel_grid.shape, dtype=np.float32), np.zeros(
        voxel_grid.shape, dtype=np.float32)

    data_left[pc_left[:, 0], pc_left[:, 1], pc_left[:, 2]] = 1.
    data_right[pc_right[:, 0], pc_right[:, 1], pc_right[:, 2]] = 1
    return data_left, data_right


def get_pca(point_cloud):
    """

    :param point_cloud: (n, n_params) ndarray
    :return: PCA fitted object from sklearn.decomposition
    """
    pca = PCA(n_components=3)
    pca.fit(point_cloud)
    return pca


def print_pca_axis(pca, point_cloud, label=""):
    """
    Display main axis of a point cloud from his pca analysis using polyscope
    :param pca:
    :param point_cloud:
    :param label:
    :return:
    """
    center = point_cloud.mean(axis=0)

    line = np.linspace(0, 1, 100)

    components = np.empty((100, 3, 3))
    for i in range(3):
        vector = pca.components_[i, :].reshape(3, 1).dot((line * pca.explained_variance_[i]).reshape(1, 100))
        # print(f"explained variance : {pca.explained_variance_[i]}")
        components[:, :, i] = center + vector.T

    ps.register_point_cloud(label + "1st pc", components[:, :, 0], color=(1., 0., 0.))
    ps.register_point_cloud(label + "2nd pc", components[:, :, 1], color=(0., 1., 0.))
    ps.register_point_cloud(label + "3rd pc", components[:, :, 2], color=(0., 0., 1.))


def pca_registration(point_cloud_source, point_cloud_target):
    """
    Register two point clouds only by fitting the main principal components axis together, as well as the center of mass
    :param point_cloud_source: (n,3) ndarray
    :param point_cloud_target: (m,3) ndarray
    :return:
    """
    pca_source = get_pca(point_cloud_source)
    pca_target = get_pca(point_cloud_target)

    U = pca_target.components_
    V = pca_source.components_

    R = U.T @ V
    print(f"det Rotation :{np.linalg.det(R)}")

    transformed_source = point_cloud_source @ R.T + (point_cloud_target.mean(axis=0) - point_cloud_source.mean(axis=0))

    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, -1] = point_cloud_target.mean(axis=0) - point_cloud_source.mean(axis=0)
    transformed_source = transform_affine(transformation, point_cloud_source)
    return transformed_source, pca_source, pca_target


def display_pca_registration(source, target):
    """
    Register and display the registration of two point clouds
    :param source: (n,3) ndarray
    :param target: (m,3) ndarray
    :return:
    """
    Tsource, pca_source, pca_target = pca_registration(source, target)

    ps.init()

    len = source.shape[0]
    ps.register_point_cloud("Transformed source", Tsource[np.random.randint(0, len, 10000), :])
    ps.register_point_cloud("Target", target)
    # ps.register_point_cloud("Source", source[np.random.randint(0, len, 10000), :])
    print_pca_axis(pca_source, Tsource, "source")
    print_pca_axis(pca_target, target, "target")
    ps.show()


def cpd_registration(source, target, mod='rigid', w=0., verbose=False, max_iterations=None):
    """
    Try to register twi point cloud using a coherent point drift algorithm
    :param source: transformed point cloud
    :param target: reference point cloud
    :param mod: 'rigid', 'affine' or 'deformable'
    :param w : between 0 and 1 - percentage of outliers
    """
    if verbose:
        callback = None
        pass  # display of every iteration not implemented
    else:
        callback = None

    if mod == 'rigid':
        registration = RigidRegistration(**{'X': target, 'Y': source, 'w': w, 'max_iterations': max_iterations})
    elif mod == 'deformable':
        registration = DeformableRegistration(**{'X': target, 'Y': source, 'w': w})
    elif mod == 'affine':
        registration = AffineRegistration(**{'X': target, 'Y': source, 'w': w})
    else:
        raise ValueError
    TY1, _ = registration.register(callback)

    if callback:
        plt.show()
    return TY1


def icp_registration(source, target, align=False, initial_transform=None):
    """

    :param source: point cloud to be registered
    :param target: point cloud used as reference
    :param align: bool,  if we translate the source to match the center of mass before registration
    :param initial_transform: initial transformation for ICP
    :return: registered point cloud
    """

    osource = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    otarget = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))

    threshold = 100

    osource.estimate_normals()
    otarget.estimate_normals()

    trans_init = np.eye(4)
    if align:
        trans_init[:3, -1] = target.mean(axis=0) - source.mean(axis=0)

    if initial_transform is not None:
        trans_init = initial_transform

    reg = o3d.pipelines.registration.registration_icp(
        osource, otarget, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return transform_affine(reg.transformation, source), reg.transformation


def rotation(k, n=np.array([0, 0, 1])):
    """
    from one point and a direction, generate a rotation the fit n into k, the rotation direction
    is the n x k axis
    :return:
    """
    w = np.cross(n, k)
    w = w / norm(w)
    theta = np.arccos(np.dot(n, k) / (norm(n) * norm(k)) ** 0.5)
    return Rotation.from_rotvec(w * theta).as_matrix()

