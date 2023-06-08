# SHERPA
### XAI for ERP analysis

`sherpa_explain.py` is a Python script that assists in the analysis and interpretation of EEG data, specifically 
event-related potentials (ERP) using SHAP 
(SHapley Additive exPlanations) values. The script utilizes various statistical methods to identify important time 
points and electrodes from the EEG data. In order to use SHERPA, first a model has to be trained classifying the
preprocessed EEG data into the experiment groups. A SHAP explainer has to be trained using this classification
model in order to get the SHAP values required as input for this script (see `sherpa_model.py`). 

## SHERPA and the N170

In our project, we used EEG data from a face perception experiment (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10086431/). 
In this first step, we trained a classifier to distinguish faces from the control conditions (blurred and scrambled faces)
and then developed SHERPA to find the ERP component.

Please note that the study from which the data stems includes more data than used in this project. We only used data from 
the PC condition, as we wanted to see if we could find the canonical N170 as would be expected. 

<small>Reference: Sagehorn M, Johnsdorf M, Kisker J, Sylvester S, Gruber T, Schöne B. Real-life relevant face perception is not 
captured by the N170 but reflected in later potentials: A comparison of 2D and virtual reality stimuli. Front Psychol. 
2023 Mar 28;14:1050892. doi: 10.3389/fpsyg.2023.1050892. PMID: 37057177; PMCID: PMC10086431.</small>

## Data
We are providing the SHAP values calculated as can be seen in `sherpa_model.py` to try out the 
functionality of SHERPA. The data can be downloaded from https://myshare.uni-osnabrueck.de/f/884c479f583549f594a6/?dl=1

## Functionality

SHERPA aggregates the absolute SHAP values for all electrodes in one condition and then proceeds to find local maxima
(function *find_extrema*). For the temporal coordinates of this maximum, a time window is defined (default 40ms) and the
SHAP values for each electrode are summed up to find the most important electrodes for this time window (function 
*find_electrodes*). The values are marked with their quantiles and saved into a csv file. The procedure is repeated for 
all conditions (faces, blurred faces, and scrambled faces; function *find_coordinates*). For our data, we used two peaks
in the face condition, as there was no substantial difference in height. (All functions are wrapped in the function
*windowed_shap*).

With the  *multi lineplot* a first intuition can be build on the important time points before running the analysis.


## Requirements
* Python 3.6 or higher
* Numpy
* Matplotlib
* Scipy
* Pandas
* SHAP
* Tensorflow
 

## Customization 
`sherpa_explain.py`

You can customize the order of the extremum finding function and the window size around the local extremum by modifying 
the `order` and `windowsize` variables respectively in the `windowed_shap` function call in the `main` function.

`order`: This parameter  controls the order of the extremum (maximum or minimum value) finding function. When 
detecting local maxima in a dataset, the order parameter  specifies how many points on each side of a point to use 
for the comparison to consider a point as a maximum. For instance, if order is set to 2, it means each point has two 
neighbors on both sides that it should be larger than in order to be considered a local maximum. Thus, increasing the 
order will make the algorithm more strict when detecting local maxima, considering a larger neighborhood around each 
point.

`windowsize`: This parameter most  defines the size of the time window around each local extremum detected. After a 
local maximum is found, a window of this specified size (in milliseconds, but the exact unit would depend on the 
specific EEG data and its sampling rate) is created around the maximum. The SHAP values for each electrode within this 
window are then aggregated.


## Output
`sherpa_explain.py`

The script generates several output files:
1. A line plot of the SHAP values for each target class at each time point, saved as a `.png` file in the directory specified by the `name` variable.
2. A `.csv` file for each target class containing a DataFrame with all electrodes, their SHAP values, and quantiles. The DataFrame is sorted by SHAP value in descending order.
