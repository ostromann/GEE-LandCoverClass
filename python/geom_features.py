import numpy as np
import pandas as pd
import cv2
import json
from swifter import swiftapply

IN_FILE = 'Beijing_2_UTM_geomee_export.csv'
OUT_FILE = 'Beijing_geom.csv'

#read in csv file as Pandas DataFrame
segments = pd.read_csv(IN_FILE).head(100).drop(['system:index', 'class', 'ID'], axis=1, errors='ignore')

#read tuple of coordinates from geojson
segments['tuple_coords'] = segments.apply(lambda row: tuple(map(tuple,np.array(json.loads(str(row['.geo']))['coordinates'][0]))),axis=1)

#compute the area
segments['area'] = segments.apply(lambda row: cv2.contourArea(np.array(row['tuple_coords'], dtype=np.int32)), axis=1)

#fit a minimum enclosing circle
# Note: due to the cast from floats to integer when reading the coordinates, 
# there is a small deviation between the area calculated by GEE and the area calculated by openCV!
segments['minEnclosingCircle_radius'] = segments.apply(lambda row: cv2.minEnclosingCircle(np.array(row['tuple_coords'], dtype=np.int32))[1], axis=1)
segments['circular_fit'] = segments.apply(lambda row: row['area']/(np.pi*row['minEnclosingCircle_radius']**2), axis=1)

#fit a minimum bounding rectangle
segments['minAreaRect'] = segments.apply(lambda row: cv2.minAreaRect(np.array(row['tuple_coords'], dtype=np.int32)), axis=1)
segments['minAreaRect_width'] = segments.apply(lambda row: row['minAreaRect'][1][0], axis=1)
segments['minAreaRect_height'] = segments.apply(lambda row: row['minAreaRect'][1][1], axis=1)
segments['minAreaRect_angle'] = segments.apply(lambda row: row['minAreaRect'][2], axis=1)

segments['minAreaRect_aspectRatio'] = segments.apply(lambda row: min(row['minAreaRect_width']/row['minAreaRect_height'], 
                                                                     row['minAreaRect_height']/row['minAreaRect_width']), 
                                                     axis=1)

segments['rectangular_fit'] = segments.apply(lambda row: int(0) 
                                             if row['minAreaRect_width']*row['minAreaRect_height'] == 0 
                                             else row['area']/(row['minAreaRect_width']*row['minAreaRect_height']), 
                                             axis=1)

#fit an ellipse
segments['fitEllipse'] = segments.apply(lambda row: cv2.fitEllipse(np.array(row['tuple_coords'], dtype=np.int32)), axis=1)
segments['fitEllipse_width'] = segments.apply(lambda row: row['fitEllipse'][1][0], axis=1)
segments['fitEllipse_height'] = segments.apply(lambda row: row['fitEllipse'][1][1], axis=1)
segments['fitEllipse_angle'] = segments.apply(lambda row: row['fitEllipse'][2], axis=1)
segments['fitEllipse_aspectRatio'] = segments.apply(lambda row: int(0) 
                                                    if row['fitEllipse_width']*row['fitEllipse_height'] == 0  
                                                    else 
                                                    min(row['fitEllipse_width']/row['fitEllipse_height'],
                                                        row['fitEllipse_height']/row['fitEllipse_width']), 
                                                    axis=1)

segments['elliptic_fit'] = segments.apply(lambda row: int(0) 
                                          if row['fitEllipse_width']*row['fitEllipse_height'] == 0
                                          else row['area']/(np.pi * row['fitEllipse_width']/2 * row['fitEllipse_height']/2), 
                                          axis=1)

#drop unnecessary columns and save to CSV file
segments.drop(['area','.geo', 'tuple_coords', 'minAreaRect', 'fitEllipse'], axis=1).to_csv(OUT_FILE, index=False)
