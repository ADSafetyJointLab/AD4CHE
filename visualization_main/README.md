## AD4CHE Visualize (The visualizer is referenced from the HighD dataset)
The python toolbox gives the opportunity to read in the AD4CHE csv files and visualize them in an interactive 
plot. Through modularity, one can use the i/o functions directly for own usage. In the following, 
each method is shortly described to maintain easy and correct usage.

```
 |-data
 |-src
    |- data_management
        |- read_csv.py
    |- utils
        |- plot_utils.py
    |- visualization
        |- visualize_frame.py
    |- main.py
```

## Quickstart
1) Copy the csv files into the **data** directory in a sub folder 
3) (Optional) Modify the folder_name (sub folder) and video_name variable in main.py or by changing 
the arguments when calling the python function
4) Run main.py

## Method descriptions
One can find short descriptions of each method implemented in this toolbox. 
### main.py
The main file is the starting point for the program. In this main file, the program first reads in 
several input arguments that control the program. There are mandatory parameter defined like the paths for the 
AD4CHE csv files. You can find the other parameters and their descriptions in the "Parameter" section below. The main file
reads in the AD4CHE data and passes it to the visualization program. The visualization program is an interactive plot that
is able to display the lanes and the vehicles driving on that lanes. One can interactively switch between frames to see 
how the tracks of the vehicles evolve over time. 
### read_csv.py
The "read_csv.py" file contains the methods for reading in the AD4CHE data. The first method "read_track_csv"
reads the information for each tracked vehicle. Every unique track contains information and position of the 
tracked and detected vehicle for each frame. 

The method "read_static_info" extracts the static information of each track. Static information is, for instance, the
direction, average velocity and more characteristic values. 

The method "read_video_meta" reads the general meta information of the whole video. 

### visualize_frame.py
The visualization program is a class "VisualizationPlot" that takes the information of the three AD4CHE csv files to create
an interactive plot that allows switching between frames of the video containing virtual vehicles. The virtual vehicles 
contain some information about their tracks, which can be shown by clicking on the corresponding yellow information box 
above each vehicle bounding box. 

### plot_utils.py
The plot utilities contain a utility function that helps to visualize the frame bar of the visualization plot.
