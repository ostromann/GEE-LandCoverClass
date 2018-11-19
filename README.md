# GEE-LandCoverClass
Object-based land cover classification with Support Vector Machines and Feature Selection for Google Earth Engine, Google Compute Engine and Scikit-Learn.

This code was developed under this Master thesis:
http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-238727


These snippets are working with the exported results of these Google Earth Engine scripts: https://earthengine.googlesource.com/users/stromann/LandCoverClass/

The GEE assets for the two case studies of Stockholm and Beijing can be found here:
* https://code.earthengine.google.com/?asset=users/stromann/LandCoverClass/Stockholm/public
* https://code.earthengine.google.com/?asset=users/stromann/LandCoverClass/Beijing/public


Input - Output
---
Input to the GEE scripts:
* Segment Polygons (as asset)
* Reference Points (as asset)

Output from GEE scripts:
* CSV files for Labels and features

Input to the Python scripts:
* Labels (csv file [segment_ID, class])
* Features (csv file [segment_ID, features])

Output from the Python scripts:
* cross-validation results (table and graphs)
* best found classifiers (*.pkl (Scikit-learn persistent model)
* learning curves
* feature importance ranking
* confusion matrices
* training and prediction times
* full land cover prediction

The full land cover prediction can be joined back to the segments in GEE for production of a land cover map.

Dependencies
---
* sklearn
* scipy
* numpy
* pandas
* matplotlib
* statsmodels
* openCV

Comments
---
Status in October 28, 2018:
I'm planning to restructure these files and provide an overview of the functions within the next months, latest with the publication in Remote Sensing of Environment's special issue on Google Earth Engine
