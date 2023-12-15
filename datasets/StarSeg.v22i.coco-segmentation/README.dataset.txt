# StarSeg > 2023-08-07 7:47pm
https://universe.roboflow.com/willem-weertman/starseg

Provided by a Roboflow user
License: Public Domain

This project is a product of the Hodin Lab at the University of Washington Friday Harbor Laboratories. It has the goal of developing a Pacific Northwest sea star instance segmentation and classification tool. Ultimately this tool will be used as a part of a sea star photo re-identification pipeline. Additionally, we hope this tool aids in the use of camera transect surveys of marine habitat.

The primary target species for this model is the Sunflower Seastar *Pycnopodia helianthoides*, the inclusion of other species is to make the model more robust to confusion species when deployed. For this reason, and our labs access to images of the Sunflower seastar it is over represented in the dataset.

If you have images of sea stars and wish to contribute to the project contact Willem @ willemlw@uw.edu

Our target number of annotated images is >10k with >100 annotated examples for each species. 

We hope to have a future extension of this model which includes both star and prey annotations.

We are drawing images from a diverse set of sources including.
	iNaturalist
	Google search
	Collaborators
	Personal lab + field images
	boldsystems

We are using the annotation tool [CVAT](https://www.cvat.ai/)

Involved members are:
Willem Lee Weertman
Marilyn Duncan
Ian Taylor
Jason Hodin