# Data Setup

## Argoverse-HD

Simply download the full dataset from the [project page](https://www.cs.cmu.edu/~mengtial/proj/streaming/) and unzip it!


### Already have Argoverse 1.1?

If you already have Argoverse 1.1, then you only need to download our small annotation package from the [project page](https://www.cs.cmu.edu/~mengtial/proj/streaming/).

Organize the images from Argoverse 1.1 into the following structure:
<p align="center"><img alt="File structure" src="img/Argoverse_file_structure.png"></p>
The hash strings represent different video sequences in Argoverse, and `ring_front_center` is one of the sensors for that sequence. Argoverse-HD annotations correspond to images from this sensor. Information from other sensors (other ring cameras or LIDAR) is not used, but our framework can be also extended to these modalities or to a multi-modality setting.

Unzip `Argoverse-HD.zip` into the same folder as Argoverse:
<p align="center"><img alt="File structure" src="img/ArgoverseHD_file_structure.png"></p>


### Notes on the annotation format
Both the annotations and the pseudo ground truth are stored in COCO format (json files). We include additional metadata to mark which image belongs to which video sequence.

Aside from the bounding box annotation, we also have a <em>track id</em> for each object (not used in this project), which can facilitate the study of object tracking. These track ids are manually annotated. However, some tracks might be broken into small segments due to the fact that the dataset is annotated in small segments with a max length of 200 frames (e.g. a 20s video is divided into 3 segments). We have attempted to link the tracks over the segment boundary and is able to resolve most of the boundary issues. Nevertheless, the track ids might still contain some annotation noise.


## Testing on a folder of images (custom data)

- Prepare a folder of images (need to be of the same size, e.g., extracted from a video)
- Use `dbcode/db_from_img_folder.py` to create a dataset meta file.
- Replace the image and annotation paths for Argoverse-HD with that for this newly created dataset in the scripts you plan to run.
