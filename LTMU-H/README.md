# Implementation of the LTMU-H tracker
This folder contains the Python implementation of the LTMU-H proposed in the paper "Visual Object Tracking in First Person Vision" appearing in IJCV.

## Authors
Matteo Dunnhofer (1)
Antonino Furnari (2)
Giovanni Maria Farinella (2)
Christian Micheloni (1)

* (1) Machine Learning and Perception Lab, University of Udine, Italy
* (2) Image Processing Laboratory, University of Catania, Italy

**Contact:** [matteo.dunnhofer@uniud.it](mailto:matteo.dunnhofer@uniud.it)


## Citing
When using the code, please reference:

```
@Article{TREK150ijcv,
author = {Dunnhofer, Matteo and Furnari, Antonino and Farinella, Giovanni Maria and Micheloni, Christian},
title = {Visual Object Tracking in First Person Vision},
journal = {International Journal of Computer Vision (IJCV)},
year = {2022}
}

@InProceedings{TREK150iccvw,
author = {Dunnhofer, Matteo and Furnari, Antonino and Farinella, Giovanni Maria and Micheloni, Christian},
title = {Is First Person Vision Challenging for Object Tracking?},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
month = {Oct},
year = {2021}
}
```

## Instructions 

1. Download the repository
   
    ```
    git clone https://github.com/matteo-dunnhofer/fpv-tracking-baselines
    cd fpv-tracking-baselines/LTMU-H
    ```

2. Create the Conda environment and install dependecies
   
    ```
    conda env create -f environment.yml
    pip install -f requirements.txt
    conda activate ltmuh
    ```

3. Download the [Hands-in-Contact repository](https://github.com/ddshan/hand_object_detector)
   
    ```
    git clone https://github.com/ddshan/hand_object_detector.git
    ```
    Download the [pretrained model for egocentric data](https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE) an put into a ```hand_object_detector/models/res101_handobj_100K/pascal_voc``` folder.

4. Download the [LTMU repository](https://github.com/Daikenan/LTMU)
    ```
    git clone https://github.com/Daikenan/LTMU.git
    ```
    Set ```base_path = './LTMU' ``` in the ```LTMU/DiMP_LTMU``` folder.

5. Download the [STARK repository](https://github.com/researchmm/Stark)

    ```
    git clone https://github.com/researchmm/Stark.git
    ```
    Then run 
    ```
    python Stark/tracking/create_default_local_file.py --workspace_dir Stark/ --data_dir Stark/data --save_dir Stark/
    ```
    Download the [baseline pretrained model](https://drive.google.com/drive/folders/1fSgll53ZnVKeUn22W37Nijk-b9LGhMdN?usp=sharing) an put into a ```Stark/checkpoints/train/stark_st2``` folder.

6. Install the TREK-150 toolkit and download the dataset
    
    Follow the instructions given at [```https://github.com/matteo-dunnhofer/TREK-150-toolkit```](https://github.com/matteo-dunnhofer/TREK-150-toolkit) to download the TREK-150 benchmark and install the evaluation toolkit.

7. Run the evaluation

    Run the evaluation on TREK-150 by running the following command.
    ```
    python evaluate_trek150.py
    ```