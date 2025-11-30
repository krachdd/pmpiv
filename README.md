# pmpiv
A `python3` based **P**orous **M**edia **P**article **I**mage **V**elocimetry toolbox for microfluidic domains developed at [MIB](https://www.mib.uni-stuttgart.de/) and [Porous Media Lab](https://www.mib.uni-stuttgart.de/pml/). 


This small and simple `python` module to provides tools to track [PIV](https://en.wikipedia.org/wiki/Particle_image_velocimetry) particles and evalute their properties. It is both, a wrapper and an extension of [`trackpy`](https://github.com/soft-matter/trackpy/tree/master) also using [`PIMS`](https://soft-matter.github.io/pims/v0.6.1/) (Python IMage Sequence) for convenience.

## Requirements
Only tested on linux desktop computers. We use `python3` and [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) to set up the virtual environments. Of course the python [`venv` module](https://docs.python.org/3/library/venv.html) is an equally suitable option. Requirements (for conda) are defined in the bash files in `envs`.

The software is specially tailored to our devices and therefore cannot be reused directly without limitations. 

Add the path to `PYTHONPATH` env variable by 
```bash
source addpythonpath.sh
```


## Input Parameters

Define your input parameters in a `.txt` file and pass it as the file path as a command line argument. 

```bash
python3 pmpiv_test.py example/pmpiv_test.txt
```
The parameters in the input text file are explained below. For even more details on specific parameters see also the `trackpy` [documentation](https://soft-matter.github.io/trackpy/v0.6.4/).

| **Parameter** 		| **Type** | **Description**                   |
|-----------------------|----------|------------------------------------|
| IN_PATH            	| str      | Folder where the input data can be found. 	|
| WORKING_DIR         	| str      | Folder to store data and save images and output in. 	|
| IN_FORMAT           	| str      | File format of the images see also documentation of [pims](https://soft-matter.github.io/pims/v0.6.1/) for more information. 	|
| PIXELSIZE         	| float    | Size of pixel in meter. 	|
| HEIGHT        		| float    | Height of the model in meter.	|
| START_FRAME        	| int      | First frame of the image sequence to be used.	|
| END_FRAME        		| int      | Last frame of the image sequence to be used. If 0 than use the whole image sequence.	|
| RATE               	| int      | If every second frame should be used -> 2. 	|
| FEATURE_SIZE       	| int      | Estimate the size of the features in pixels, must be ODD integer. 	|
| FEATURE_MIN_SIZE      | int      | Threshold parameter in pixels, must be odd integer. If image is especially noisy. 	|
| FEATURES_ARE_DARK   	| bool     | If features in images are dark, set to true. 	|
| FPS               	| float    | Frames per second [1/s]. 	|
| MAX_PARTICLE_SPEED   	| int      | Maximum displacement, the farthest a particle can travel between frames.  We should choose the smallest reasonable value because a large value slows computation time considerably. 	|
| MEMORY            	| int      | There is the possibility that a particle might be missed for a few frames and then seen again. Memory keeps track of disappeared particles and maintains their ID for up to some number of frames after their last appearance.|
| DURATION          	| int      | Remove ephemeral trajectories that last shorter than DURATION frames. |
| JSON_PATH            	| str      | Folder where the json files with COCO annotations can be found. 	|
| REMOVAL            | str      | Comma seperated list of strings. json files with annotations to delete particles within these regions. 	|
| EXTRACTION             | str      | Comma seperated list of strings. json files with annotations to select particles within these regions. 	|




## Program Flow 

1. Define Input parameters
2. Compute Sequence statistics: locates all features in some frame and computes stats such as size, eccentricity
3. Compute histograms for relevant feature charcteristics
4. Locate Gaussian-like blobs and store all info for all frames in DF
5. Percentile filtering (optional), e.g. remove all features with mass in top or lowest 3% of masses
6. Link particles with Crocker-Grier linking algorithm. Compute trajectories.
7. Remove spurious trajectories with a duration < given threshold.
8. Create Annotations for solid structures.
9. Read Annotation and remove 
10. Vice versa annotations can be used to extract subsets. 
11. Compute area, volume and average particle velocity for subsets. 
13. Compute particle density per subset.


## Annotations

To annotate areas of interest or to neglect and export the data as `.json` files one may use the [Image-Annotator](https://github.com/bnsreenu/digitalsreeni-image-annotator). If other tools are used, the class `Annotation_Filtering` (in `src/filtering.py`) may have to be adapted. 

Use the annotator as follows:
```bash
conda activate image_anno
digitalsreeni-image-annotator
```
An overview of the necessary `conda` packages for the annotation tool can be found on the github page linked above or the bash script.

## Features 

#### Remove or Extract annotated regions
List the `.json` files with annotations in the input file. 


## Acknowledgements
Funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy (Project number 390740016 - EXC 2075 and the Collaborative Research Center 1313 (project number 327154368 - SFB1313). We acknowledge the support by the Stuttgart Center for Simulation Science (SimTech).

## Developer
- [David Krach](https://www.mib.uni-stuttgart.de/institute/team/Krach/) E-mail: [david.krach@mib.uni-stuttgart.de](mailto:david.krach@mib.uni-stuttgart.de)


## Contact
- [Software Support Institute of Applied Mechanics](mailto:software@mib.uni-stuttgart.de)
- [Data Support Institute of Applied Mechanics](mailto:data@mib.uni-stuttgart.de)
