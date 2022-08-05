# Registration of thyroid lobes

Project done in order to be able to register US image of left and right lobe of the thyroid. Prior to this work is that we already have some segmentation mask.

# library used :

numpy 
scipy
scikit-learn
circle-fit (https://github.com/AlliedToasters/circle-fit)
matplotlib
trimesh (for smoothing mesh)
scikit-image

pandas
seaborn
polyscope (to generate visual of thyroids)
nibabel (to load nifti images)
pathlib
open3d (for a few functionnalities: merging two mesh, generate a mesh of a cylinder, not mandatory)



The main algorithm is in cylinder_detection.py : the main function is cylinder_detection that detect a cylinder on the surface of a mesh using prior information.

main.py is an example of how to use the algorithm

registration_and_analysis.py generate registration of all volunteer data, and store the results in a specified output file. Be carefull, because I'm computing the algorithm separately on each lobe, and on both lobes merged together (in order to compare my registration and some previous GT atlas)

create_atlas_ICP.py was written to generate this GT atlas from an ICP algorithm an thyroid surface mesh.

result_analysis.ipynb was used to analyze data generated by registration_and_analysis.py 


In functions.py are some usefull function for loading nifti files, as well as generating mesh with them. Few other transformation are coded here (in order to manipulate point cloud). At the end of the file are few functions used to generate an ICP atlas from MRI to US. These are not mandatory for other algorithm.
