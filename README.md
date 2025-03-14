# NeurErrors -- WIP

This repository contains a Python class, `Dataset_Client`, designed to acquire and build graph datasets for past segment IDs in a chunked graph. The class provides functionalities for associating error coordinates with graph datasets and visualizing them in Neuroglancer.


## What this can do

- Acquire a graph dataset for a given segment ID
- Associate error coordinates with a graph dataset
- Visualize localization of errors in neuroglancer

Results: 

We can also associate errors with a given segment ID and visualize them in neuroglancer:
![alt text](./images/results_url.png)

At the L2 resolution, we can have the graph visualization of the following:
![alt text](./images/graph_visu.png)



## Installation

You need to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
pip install git+https://github.com/neurodata/neurerrors.git #TODO: correct link
```

An example of how to use the `Dataset_Client` class is provided in the `example.ipynb` file.

You can do:



## Acknowledgements

I would like to thank Sebastian Seung and the whole PNI and FlyWire team for their help and feedback on this project.




