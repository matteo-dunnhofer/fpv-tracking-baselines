# Implementation of the TbyD-H tracker
This folder contains the Python implementation of the TbyD-H proposed in the paper "Visual Object Tracking in First Person Vision" appearing in IJCV.

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
    cd fpv-tracking-baselines/TbyD-H
    ```

2. Download Hands-in-Contact

    Follow the instructions given at [```https://github.com/ddshan/hand_object_detector```](https://github.com/ddshan/hand_object_detector) to download and install the Hands-in-Contact detector. Download the [pretrained model for egocentric data](https://drive.google.com/open?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE) as stated in the original repo.

3. Install the TREK-150 toolkit and download the dataset
    
    Follow the instructions given at [```https://github.com/matteo-dunnhofer/TREK-150-toolkit```](https://github.com/matteo-dunnhofer/TREK-150-toolkit) to download the TREK-150 benchmark and install the evaluation toolkit.

4. Run the evaluation

    Run the evaluation on TREK-150 by running the following command.
    ```
    conda activate handobj
    python evaluate_trek150.py
    ```