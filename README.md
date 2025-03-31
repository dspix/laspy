# Evidence for critical slowing down and transitions to alternative stable states in ecosystems

Repository for the wavelet-based filtering of satellite time series signals (Selective Wavelet Extraction for the Estimation of Phenology, SWEEP) and visualization of *stability landscapes* for measuring ecosystem change trajectories at UK reference sites (COSMOS-UK) that evidence dynamic ecological regimes.

## Installation

Code should run on Linux, OSX, and Windows operating systems. Tested on Ubuntu 22.04.4 LTS and Windows 10 Enterprise Build 19045.

Requires Python 3 and [Jupyter](https://jupyter.org/) to run the example *Notebook*. Recommended to first install [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/main) for [your operating system](https://www.anaconda.com/docs/getting-started/anaconda/install) and install missing [dependencies](https://github.com/dspix/laspy/blob/main/requirements.txt) using `conda install` or `pip`, e.g.

```bash
pip install -r requirements.txt.
```

Clone this repository and add the `source` folder to the Python path to run the project code.

## Demo

To run the example [example *Notebook*](https://github.com/dspix/laspy/blob/main/notebooks/sweep_landscapes_example.ipynb), first start a Jupyter server in the cloned repo (e.g. `$ jupyter notebook` or `$ jupyter lab`) and open the Notebook `sweep_landscapes_example.ipynb` using the browser interface. The code should run in less than a minute on a normal desktop/laptop computer.

The example can be used to reconstruct all the analysis in the paper (link TODO).

## Running on your data

You can run SWEEP for any timeseries NDVI dataset using the demo as a template. For the DLM you need to provide climate variables for the location, see the format of the example dataset ([`glenwherry_example.csv`](notebooks/example_data)) and the preprocessing in the example for details.
