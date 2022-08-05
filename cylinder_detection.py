from sklearn.decomposition import PCA
import circle_fit as cf
import numpy as np
from numpy.linalg import norm

from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
from scipy import signal
from functions import transform_affine
from trimesh.intersections import mesh_multiplane
import trimesh
from skimage.feature import peak_local_max

import os


def cylinder_detection(point_cloud_3d, faces, recompute_normals=False, verbose=True,
                       iter=3,
                       **kwargs):
    """
    Detect a cylinder among a point cloud

    :param point_cloud_3d: (n,3) ndarray float  - vertices of the thyroid mesh
    :param faces: (n,3) ndarray int - faces of thyroid's mesh
    :type recompute_normals: bool - if normals should be recomputed to be smoother - strongly advised when using a
        slice per slice segmented thyroid
    :param verbose: bool - if log should be printed or not

    :return:
        center: (3,) ndarray
        direction: (3,) ndarray
        radius: float
        length: float
        localization_map: (n,) ndarray
    """
    date = time.time()

    localization_treshold = 0.9

    n = point_cloud_3d.shape[0]

    # Smoothing the thyroid in order to have consistent normals and regularly spaced points

    mesh = trimesh.Trimesh(point_cloud_3d, faces)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)
    mesh = trimesh.smoothing.filter_laplacian(mesh)

    point_cloud_3d = mesh.vertices
    faces = mesh.faces
    normals = mesh.vertex_normals

    # Computation of robust normals vector using information of outside / inside from "normals"
    if normals is None:
        normals = detect_normals(point_cloud_3d, normal_radius=1., verbose=verbose)
    elif recompute_normals:
        normals_ = detect_normals(point_cloud_3d, normal_radius=1., verbose=verbose).astype(normals.dtype)
        scalar_product = (normals * normals_).sum(axis=1)
        normals = (normals_.T * (scalar_product.T >= 0) - normals_.T * (scalar_product.T < 0)).T

    # Core of the algorithm : loop of Hough transform and Ransac vote

    localization_map = np.ones(n)
    best_cylinder = None

    for i in range(iter):
        if 'callback' in kwargs.keys():
            kwargs['callback'].new_iter()

        localization_index = np.argwhere(localization_map > localization_treshold)[:, 0]
        possible_directions, peak_values = hough_orientation_detection(precision=1,
                                                                       normals=normals[localization_index, :],
                                                                       pike_treshold=0.8,
                                                                       smoothing_kernel=2.,
                                                                       verbose=verbose,
                                                                       **kwargs)
        if verbose:
            print(f"{len(possible_directions)} best directions selected")

        delta_map, best_cylinder = ransac_fit(point_cloud_3d=point_cloud_3d,
                                              faces=faces, normals=normals,
                                              possible_directions=possible_directions,
                                              direction_likelihood=peak_values,
                                              verbose=verbose, **kwargs)

        localization_map += delta_map
        localization_map = localization_map / localization_map.sum() * localization_map.size

    r = best_cylinder[1]
    axis = best_cylinder[2]
    length = (point_cloud_3d @ axis).max() - (point_cloud_3d @ axis).min()
    cylinder_center = best_cylinder[0] + (
            ((point_cloud_3d @ axis).max() + (point_cloud_3d @ axis).min()) / 2 - best_cylinder[0] @ axis) * axis

    if verbose:
        print(f"cylinder detected : (0,n,r,L) = {cylinder_center},{axis},{r},{length}")
        print(f"cylinder detection done in {time.time() - date:.3f}s")

    return cylinder_center, axis, r, length, localization_map


def ransac_fit(point_cloud_3d, faces, normals, possible_directions, direction_likelihood=None, slice_step=1,
               verbose=False, **kwargs):
    """
    :param point_cloud_3d: (n,3) ndarray - point cloud of the thyroid to fit a circle on
    :param normals: (n,3) ndarray - 3d normals of the point cloud
    :param faces: (n,3) int ndarray - faces of the thyroid mesh
    :param slice_step: float - steps between two slice
    :param possible_directions: ndarray(3,) list - possible directions of the trachea
    :param direction_likelihood: float list - likelihood of the direction in [0,1]
    :return: (n,) ndarray, (0,d,r) - localization map variation and best detected cylinder
    """
    date = time.time()

    # Hyper parameters :
    random_sample_tot = int(np.log(1 - 0.95) / np.log(1 - 0.25 ** 3))  # maximum iterations allowed
    ransac_threshold = 2.  # threshold value to determine if points fit well or not (distance in mm)
    rmin, rmax = 5., 12.5  # normal trachea diameter is in [10mm, 25mm]
    localization_map = np.zeros(point_cloud_3d.shape[0])
    best_cylinder, best_loss = None, np.infty

    if verbose:
        print(f"Ransac loop in {random_sample_tot} steps... ")

    for i_direction, direction in enumerate(possible_directions):
        if 'callback' in kwargs.keys():
            kwargs['callback'].new_direction()

        # Generating different slices of the thyroid for RANSAC fitting:
        z_coordinate = point_cloud_3d @ direction
        length = z_coordinate.max() - z_coordinate.min()
        min_z = z_coordinate.argmin()
        plane_origin = point_cloud_3d[min_z]
        slice_steps = np.arange(start=2 * slice_step, stop=length - 2 * slice_step, step=slice_step)
        mesh = trimesh.Trimesh(vertices=point_cloud_3d, faces=faces)
        lines, to_3d, face_index = mesh_multiplane(mesh, plane_origin=plane_origin,
                                                   plane_normal=direction, heights=slice_steps)
        slices = [line.mean(axis=1) for line in lines]

        for i_slc, slc in enumerate(slices):
            if slc.shape[0] == 0:
                print("no point in slice !")
                continue
            slice_normals = project_normals(face_index[i_slc], direction, to_3d[i_slc], normals, faces)

            k = random_sample_tot * slc.shape[0] // point_cloud_3d.shape[0] + 1
            min_compatible_data = 0.001 * slc.shape[0]  # Number of close data points to assert model fits well

            possible_circles = np.empty((k, 3))  # x, y, r

            date_ransac = time.time()
            i: int = 0
            while i < k:
                if (time.time() - date_ransac > 0.25):
                    i = i+1
                    date_ransac = time.time()

                # Random model
                points_id = np.random.choice(slc.shape[0], 3, replace=False)
                points_selection = slc[points_id]

                if (norm(points_selection[0, :] - points_selection[1, :]) < 1e-6 or
                        norm(points_selection[0, :] - points_selection[2, :]) < 1e-6 or
                        norm(points_selection[1, :] - points_selection[2, :]) < 1e-6):
                    continue
                try:
                    x, y, r = excircle(points_selection)
                except RuntimeError:  # When 3 points are aligned, excircle can't fit a circle to them
                    continue



                O = np.array([x, y])
                distances = norm(slc - O, axis=1) - r
                compatible_points = slc[np.argwhere(
                    (distances < ransac_threshold) *
                    (((slc - O) * slice_normals).sum(axis=-1) <= 0)
                )[:, 0]]

                if (compatible_points.shape[0] > min_compatible_data) and \
                        (((points_selection - O) * slice_normals[points_id, :]).sum(axis=-1) <= 0).all() and \
                        rmin < r < rmax:
                    x2, y2, r2, l2 = cf.hyper_fit(compatible_points)
                    possible_circles[i, :] = np.array([x2, y2, r2])
                    i += 1

            centers_2d = np.zeros((possible_circles.shape[0], 3))
            centers_2d[:, :2] = possible_circles[:, :2]

            centers = transform_affine(to_3d[i_slc], centers_2d)
            radius = possible_circles[:, 2]

            for j in range(k):
                loss, delta_mu = ransac_loss(
                    cylinder=(centers[j, :], radius[j], direction),
                    point_cloud_3d=point_cloud_3d, threshold=ransac_threshold,
                    normals=normals,
                    direction_likelihood=direction_likelihood[i_direction]
                )
                # print_circle(possible_circles[j,:2], radius[j], slc, title=f"circle fit with loss {loss}")
                localization_map += delta_mu

                if 'callback' in kwargs.keys():
                    kwargs['callback'].add_entry(centers[j], radius[j], 1 - loss,
                                                 spherical_coordinates(direction.reshape((1, 3)))[0, :] / np.pi * 180)

                if loss < best_loss and \
                        direction_likelihood[
                            i_direction] > 0.99 or best_loss * 0.9 > loss:  # high confidence needed for direction
                    best_loss = loss
                    best_cylinder = (centers[j, :], radius[j], direction)
                    # print_circle(possible_circles[j,:2], radius[j], slc, title=f"circle fit with loss {loss}")

    return localization_map, best_cylinder


def ransac_loss(cylinder, point_cloud_3d, normals, threshold, direction_likelihood):
    """
    :param direction_likelihood:
    :param cylinder: (0,r,d) params of the cylinder
    :param point_cloud_3d: (n,3) : point cloud
    :param normals: (n,3) : normals of the point_cloud
    :param threshold: float : define the points in the inlier subset
    :return: loss, delta_mu : loss corresponding to the cylinder and a localization map with it
    """
    mu = 10
    sigma = 4
    epsilon = threshold / 1.96

    O, r, d = cylinder

    point_cloud_2d = point_cloud_3d - O
    point_cloud_2d = point_cloud_2d - np.outer(point_cloud_2d @ d, d)
    distances = norm(point_cloud_2d, axis=-1)
    inlier_index = np.argwhere(
        1 -
        (1 - (distances < r + threshold) * ((point_cloud_2d * normals).sum(axis=-1) <= 0))
        * (distances > r)
    )[:, 0]  # inliers are closer than r or closer than r + threshold AND normal is going towards center of the circle

    inlier_points_2d = point_cloud_2d[inlier_index]
    distances_inlier = distances[inlier_index]

    l_compatibility = (inlier_points_2d.shape[0] / point_cloud_3d.shape[0])  # in [0,1]
    l_least_square = np.exp(-0.5 * ((distances_inlier - r) ** 2).mean() / epsilon ** 2)
    l_prior = np.exp(-0.5 * (r - mu) ** 2 / sigma ** 2)

    loss = 1 - (l_prior * l_compatibility * l_least_square * direction_likelihood)
    delta_mu = np.zeros((point_cloud_3d.shape[0],))
    delta_mu[inlier_index] += 1 - loss

    return loss, delta_mu


def project_normals(faces_index, plane_normal, affine_to_3d, mesh_normals, mesh_faces):
    """
    Detect the normals in slices from the normals of the points in the 3D point cloud
    :param faces_index: ndarray (m,) - index of the faces intersected by the slice
    :param plane_normal: normal to the slice
    :param affine_to_3d: ndarray (4,4) - affine transformation transforming the slice from 2d to 3d
    :param mesh_normals: ndarray (n,3) - 3d normals of the thyroid
    :param mesh_faces: ndarray (n,3) - faces of the thyroid mesh
    :return: (n,2) - normals projected in the slice
    """
    faces_selection = mesh_faces[faces_index]
    lines_normal_3d = (mesh_normals[faces_selection[:, 0]] +
                       mesh_normals[faces_selection[:, 1]] +
                       mesh_normals[faces_selection[:, 2]]) / 3

    lines_normal_3d_projected_on_plane = lines_normal_3d - np.outer((lines_normal_3d @ plane_normal), plane_normal)
    lines_normal_2d = (lines_normal_3d_projected_on_plane @ affine_to_3d[:3, :3])
    return lines_normal_2d[:, :2]


def hough_orientation_detection(point_cloud3D=None, normals=None, precision=1.,
                                pike_treshold=0.7, smoothing_kernel=1., verbose=True, **kwargs):
    """
    Detect the orientation a point-cloud sampled around a cylinder

    :param point_cloud3D: (n,3) ndarray - point_cloud of the surface of the cylinder. Either this or normals should not be None
    :param precision: float - angle precision of the discretization (in °)
    :param normals: (n,3) ndarray - normals vector of the point cloud
    :param smoothing_kernel: float - in °, gaussian kernel used to blur the hough space
    :param pike_treshold: float - threshold to select local pike in the hough transform, relative to the max
    :param verbose: bool - print intermediate result or not
    :return: ndarray (3,1) list - list of most probable directions, probability associated
    """

    """
    We use for the Hough spherical space a unorthodox parametrization : 
    (x,y,z) = (cos(phi),sin(phi)cos(theta),sin(phi)sin(theta))
    the goal is to have a quasi isotropic and smoother grid around the z axis 
    (cylinder-axis is near z axis as prior)
    """
    angle_precision = precision / 180 * np.pi
    date = time.time()

    if normals is None:
        normals = detect_normals(point_cloud3D, normal_radius=0.6)

    # creation of hough space

    hough_sphere_normals = spherical_coordinates(normals)
    if verbose == 2:
        plt.scatter(hough_sphere_normals[:, 0], hough_sphere_normals[:, 1], alpha=0.003)
        plt.title(f"normals directions")
        plt.xlabel("theta")
        plt.ylabel("phi")
        plt.show()

    # hough votes stage
    t = np.arange(0, 2 * np.pi, angle_precision * 2)
    basic_circle = np.array([np.cos(t), np.sin(t), 0 * t]).T
    if verbose:
        print(f"hough voting system: {normals.shape[0]} points to handle ...")

    # prior limit angle for orientation
    prior_angle = 90.
    theta_limit = (90 - prior_angle) / 180 * np.pi
    phi_limit = (90 - prior_angle) / 180 * np.pi

    theta = np.arange(theta_limit, np.pi - theta_limit + angle_precision, angle_precision)
    phi = np.arange(phi_limit, np.pi - phi_limit + angle_precision, angle_precision)

    # ATTENTION : May overlap current memory of too much normals vector and precision
    # Memory use : batch_size * 360 * 4 / precision bytes
    # e.g if precision ~ 0.1, batch_size = 2¹², memory use is ~1.44 Go, but actually much more like ~10 Go
    batch_size = int(2 ** 12)

    hough_vote_batch = np.zeros((batch_size, basic_circle.shape[0], 2), dtype=np.float32)
    hough_sphere_directions = np.zeros((theta.shape[0] - 1, phi.shape[0] - 1), np.float64)

    for id_point in range(normals.shape[0]):
        point = normals[id_point]
        p = np.array([0, 0, 1]) - point
        if (norm(p) != 0):
            p /= norm(p)
            rot = np.eye(3) - 2 * np.outer(p, p)
            hough_vote_batch[id_point % batch_size, ...] = spherical_coordinates(basic_circle @ rot.T).astype(
                np.float32)
        else:
            hough_vote_batch[id_point % batch_size, ...] = spherical_coordinates(basic_circle).astype(np.float32)

        if id_point % batch_size == 0 and id_point != 0:
            hough_vote_temp = hough_vote_batch.reshape((-1, 2))
            hough_directions_batch, _, _ = np.histogram2d(hough_vote_temp[:, 0],
                                                          hough_vote_temp[:, 1],
                                                          bins=[theta, phi])
            hough_sphere_directions += hough_directions_batch

        elif id_point == normals.shape[0] - 1:
            hough_vote_temp = hough_vote_batch[id_point % batch_size, ...].reshape((-1, 2))
            hough_directions_batch, _, _ = np.histogram2d(hough_vote_temp[:, 0],
                                                          hough_vote_temp[:, 1],
                                                          bins=[theta, phi])
            hough_sphere_directions += hough_directions_batch

    # Selection of the best direction

    blurred_hough_space = blur_hough_space(hough_sphere_directions, precision, kernel_size=smoothing_kernel)
    local_peak = peak_local_max(blurred_hough_space, threshold_rel=pike_treshold, exclude_border=2, num_peaks=8)

    if verbose:
        print(f"Hough transform done in {time.time() - date:0.3f}")
        if verbose == 2:
            hough_sphere_plot(blurred_hough_space, title="blurred Hough voting space", imsave=False)
            hough_sphere_plot(hough_sphere_directions, title="Hough voting space", imsave=False)
            pike_mask = np.zeros_like(blurred_hough_space, dtype=bool)
            pike_mask[tuple(local_peak.T)] = True
            hough_sphere_plot(pike_mask * blurred_hough_space, title="local pike", imsave=False)

    possible_directions = []
    for i, peak in enumerate(local_peak):

        direction_spherical = np.array([theta[peak[0]], phi[peak[1]]])

        direction = np.array([np.cos(direction_spherical[1]),
                              np.sin(direction_spherical[1]) * np.cos(direction_spherical[0]),
                              np.sin(direction_spherical[1]) * np.sin(direction_spherical[0])])

        possible_directions.append(direction)

        if verbose and i == 0:
            print(f"best direction : theta,phi : {direction_spherical * 180 / np.pi}")
            print(f"Corresponding axis : {direction}")

    return possible_directions, blurred_hough_space[local_peak[:, 0], local_peak[:, 1]] / blurred_hough_space.max()


def blur_hough_space(hough_space, precision, kernel_size=2.):
    """
    Blur the hough space
    :param hough_space: input hough vote space
    :param precision: precision of the grid vote, in °
    :param kernel_size: standard deviation of the gaussian kenrel used for blurring, in °
    :return: blurred hough space of the same size as hough_space
    """
    sigma = kernel_size / precision

    kernel = np.outer(signal.windows.gaussian(int(6 * sigma) + 1, sigma),
                      signal.windows.gaussian(int(6 * sigma) + 1, sigma))

    blurred_hough_space = signal.convolve(hough_space, kernel, mode='same')
    return blurred_hough_space


def direction_selection(hough_space, precision, kernel_size=2, method='gaussian_blur', verbose=False):
    """
    Select possible directions from hough
    :param hough_space: (N,M) ndarray
    :param precision: in °, angle precision of each cell
    :param kernel_size: kernell size of the gaussian blur, in °
    :param method: 'gaussian_blur', 'gaussian KDE' (not implemented) or 'avg'
    :return: one direction (theta, phi) from the hough space, supposed to be the most likely one
    """

    if method == 'gaussian_blur':
        blurred_hough_space = blur_hough_space(hough_space, precision, kernel_size)
        if verbose == 2:
            hough_sphere_plot(blurred_hough_space, title="Blurred Hough voting space", imsave=False)
            # hough_sphere_plot(hough_space, title="Houhh voing space", imsave=False)

        max_direction = np.argmax(blurred_hough_space)
        mean_direction = np.array([max_direction // blurred_hough_space.shape[1],
                                   max_direction % blurred_hough_space.shape[1]])
        return mean_direction

    if method == 'avg':
        mask = hough_space > np.quantile(hough_space, 0.95)
        masked_hough_space = mask * hough_space
        masked_hough_space /= np.sum(masked_hough_space)
        X, Y = np.meshgrid(np.arange(hough_space.shape[0]), np.arange(hough_space.shape[1]))
        X = (X * masked_hough_space).ravel()
        Y = (Y * masked_hough_space).ravel()
        mean_direction = np.sum(np.array([X, Y]), axis=1)
        if verbose == 2:
            hough_sphere_plot(masked_hough_space, title="masked hough space", imsave=False)
        return mean_direction[1], mean_direction[0]


def detect_normals(point_cloud3D, normal_radius=1.2, avg_nn=40, verbose=True):
    """
    Detection normals in point_cloud3D using PCA and kdT
    :param point_cloud3D: (n,3) ndarray
    :param normal_radius: float, radius
    :param avg_nn: int - average neighbors for each point to guess the normal - hypothesis is made that the density is
        around 1/0.12² point/mm²
    :return: (n,3) : normals of the point_cloud
    """
    date = time.time()
    if verbose:
        print(f"detecting normals on {point_cloud3D.shape[0]} points ...")
    kdt = KDTree(point_cloud3D)

    subsample_size = int(0.12 ** 2 * avg_nn * point_cloud3D.shape[0] / (3.14 * normal_radius ** 2))
    subsample_size = min(subsample_size, point_cloud3D.shape[0])
    print(f"subsample size : {subsample_size}, original size : {point_cloud3D.shape[0]}")
    subkdt = KDTree(point_cloud3D[np.random.choice(point_cloud3D.shape[0], subsample_size, replace=False), :])
    result = kdt.query_ball_tree(subkdt, r=normal_radius)

    normals = np.zeros((kdt.data.shape[0], 3), dtype=point_cloud3D.dtype)

    for i in range(kdt.data.shape[0]):
        pca = PCA(3)
        pca.fit(subkdt.data[result[i]])
        normals[i, :] = pca.components_[2, :] / norm(pca.components_[2, :])

    if verbose:
        print(f"Normals detection done in {time.time() - date:.3f}s")
    return normals


def hough_sphere_plot(hough_sphere_directions, title="", theta_phi=None, imsave=False):
    img = hough_sphere_directions  # *(hough_sphere_directions > np.quantile(hough_sphere_directions, 0.95))
    img_zeros = np.argwhere(img == 0)
    img[img_zeros[:, 0], img_zeros[:, 1]] = 1.
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(title)
    cmap_ = 'inferno'
    norm_ = mpl.colors.Normalize(vmin=img.min(), vmax=img.max())
    plt.imshow(img.T, norm=norm_, cmap=cmap_)
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\phi$")
    if imsave:
        dir = "/home/renaud/Pictures/hough_transform/"
        list = os.listdir(dir)
        number_files = len(list)
        print(f'image_resolution : {img.shape}')
        plt.imsave(dir + f"{number_files}.png", img.T, cmap=cmap_)

    plt.show()


def print_circle(O, r, point_cloud2D, title, imsave=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(point_cloud2D[:, 0], point_cloud2D[:, 1], color='blue', alpha=0.3)

    circle = plt.Circle(O, r, color='red', fill=False)
    ax.add_patch(circle)
    ax.set_title(title)
    plt.show()
    if imsave:
        dir = "/home/renaud/Pictures/circle_fit/"
        list = os.listdir(dir)
        number_files = len(list)
        fig.savefig(dir + f"{number_files}.png")


def excircle(points):
    """
    :return:  x0 and y0 is center of a circle, r is radius of a circle
    """
    x1, y1 = points[0, :]
    x2, y2 = points[1, :]
    x3, y3 = points[2, :]
    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    a1 = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0
    a2 = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0
    theta = b * c - a * d
    if abs(theta) < 1e-9:
        raise RuntimeError('There should be three different x & y !')
    x0 = (b * a2 - d * a1) / theta
    y0 = (c * a1 - a * a2) / theta
    r = np.sqrt(pow((x1 - x0), 2) + pow((y1 - y0), 2))
    return x0, y0, r


def spherical_coordinates(xyz):
    """
    Transform a point from cartesian to spherical coordinates
    :param xyz: (n,3) ndarray (cos(phi),sin(phi)cos(theta),sin(phi)sin(theta))
    :return: (n,2) ndarray (theta, phi)
    """
    ptsnew = np.zeros((xyz.shape[0], 2))
    ptsnew[:, 0] = np.arctan2(xyz[:, 2], xyz[:, 1])
    ptsnew[:, 1] = np.arccos(xyz[:, 0])
    return ptsnew


def cartesian_coordinates(theta_phi):
    """
    Transform a point from cartesian to spherical coordinates
    :param theta_phi: (n,2) ndarray (theta, phi)
    :return: (n,3) ndarray (cos(phi),sin(phi)cos(theta),sin(phi)sin(theta))
    """
    direction = np.zeros((theta_phi.shape[0], 3))
    direction[:, 0] = np.cos(theta_phi[:, 1])
    direction[:, 1] = np.sin(theta_phi[:, 1]) * np.cos(theta_phi[:, 0])
    direction[:, 2] = np.sin(theta_phi[:, 1]) * np.sin(theta_phi[:, 0])
    if np.isnan(direction).any():
        nan_indices = np.argwhere(np.isnan(direction)).shape[0]
        direction[nan_indices, :] = np.zeros((nan_indices.shape[0], 2))
    return direction
