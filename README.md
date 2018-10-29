# GEE-LandCoverClass
Object-based land cover classification with Support Vector Machines and Feature Selection for Google Earth Engine, Google Compute Engine and Scikit-Learn.

These snippets are working with the exported results of these Google Earth Engine scripts: https://earthengine.googlesource.com/users/stromann/LandCoverClass/

Input to the GEE scripts:
- Segmented image objects (as asset)
- Reference Points (as asset)

Output from GEE scripts:
- CSV files for Labels and features

Input to the Python scripts:
- Labels (csv file [segment_ID, class])
- Features (csv file [segment_ID, features])

Output from the Python scripts:
- cross-validation results (table and graphs)
- best found classifiers (*.pkl (Scikit-learn persistent model)
- learning curves
- feature importance ranking
- confusion matrices
- training and prediction times
- full land cover prediction

The full land cover prediction can be joined back to the segments in GEE for production of a land cover map.


Status in October 28, 2018:
Right now this repository is a loose collection of code snippets.
I'm planning to bring some order into these files and provide an overview of the functions within the next months.
