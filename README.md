# TrafficScenarios-RFAPsClustering

The official code for the paper: 

## Dependencies
- Python (>=3.6)
- scikit-learn (0.21.0) (custom) -> Installation instructions given below
- scipy (1.5.2)
- numpy (0.19.0)
- Pytorch (1.14)

## Table of Contents

1. [General descriptiopn
](#gs)
2. [Traffic Scenarios
](#ts)
3. [Clustering
](#EVT)

## General description<a name="gs"></a>

The 7 highways classes are divided into 4 labelled classes and the rest of the traffic scenarios from the three classes are considered unlabelled. The clustering is done in a three step process. Step I: Self-supervised initialisation, a 3D CNN is trained on a pretext task for predicting the temporal order of the occupancy grids representing the traffic scenarios, Step II: Classification, the trained 3D CNN is fine-tuned to classify the 4 labelled classes, Step III: CLustering, the 3D CNN trained until now is iteratively optimised to cluster the scenarios from the 3 unlabelled classes. 

## Traffic Scenarios<a name="ts"></a>
The traffic scenarios are generated from the HighD Dataset [1]. 7 common highway scenarios are extracted from highD dataset. The 7 scenarios are as follows:

	- Ego - Following: The ego vehicle follows a leader vehicle.
	- Ego - Right lane change: The ego makes a lane change to the right lane.
	- Ego - Left lane change: The ego makes a lane change to the left lane.
	- Leader - Cutin from left: The leader vehicle  makes a lane change in front of the ego lane from the left lane of ego. 
	- Leader - Cutin from right: The leader vehicle  makes a lane change in front of the ego lane from the right lane of ego.
	- Leader - Cutout to left: The leader vehicle  makes a lane change from  the ego lane to the left lane of ego.
	- Leader - Cutout to right: The leader vehicle  makes a lane change  from  the ego lane from the right lane of ego.
	

## Clustering<a name="EVT"></a>

Please fill in the forms to request access to the HighD Data from https://www.highd-dataset.com/. The code for processing the data is available in https://github.com/lab176344/Traffic_Sceanrios-VoteBasedEVT. Process the data and place the .mat file in .data/datasets/Scenarios/ and use the name HighDScenarioClassV1_Train/Test/Val. Followed by that the scripts mentioned below can be run one after the other to cluster the scenarios.

Step 1: Self-Supervised Initialisation -> ```python selfsupervised_learning_scenario.py```

Step 2: Classification -> ```python supervised_learning_scenarios.py```

Step 3: Clustering -> ```python clustering_scenarios.py```

## Reference
[1] The highD Dataset: A Drone Dataset of Naturalistic Vehicle Trajectories on German Highways for Validation of Highly Automated Driving Systems, Krajewski et al., ITSC 2018
