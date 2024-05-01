# whole_AL_segmentation

To install all the appropriate python packages, please type the following on the command line:

`mamba create -n caiman -y anaconda::scikit-image conda-forge::caiman numpy seaborn pandas matplotlib`

If you find that running `01_segment_and_extract_traces.ipynb` takes a long time, you can run the following command:

`jupyter nbconvert --to script 01_segment_and_extract_traces.ipynb`

which converts the notebook to a python script. You can then submit a job that runs this script via `run.sh`.

The notebook `01_segment_and_extract_traces.ipynb` creates two subdirectories: 
- `binary_mask_plots`:  stores 2D projections of the binary mask that is used to compute mean fluorescence over time
- `results`: stores csv files of the extracted signals that can be loaded into DataFrames for downstream analyses