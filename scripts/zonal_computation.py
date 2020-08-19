import ee
import geemap
import os
import numpy as np
import csv
import pandas as pd
import bqplot as bq
import json
from scripts import gdrive
from sepal_ui.scripts import utils as su
from sepal_ui.scripts import gee
import glob
import gdal
from sepal_ui import gdal as sgdal
import subprocess
import gdalconst

ee.Initialize()

def getVal(feat):
    feat = ee.Feature(feat)
    vals = ee.Dictionary(feat.get('histogram')).keys()
    return ee.Feature(None, {'vals': vals})

def run_zonal_computation(country_code, Map, output):
    
    list_zones = get_ecozones()
    
    drive_handler = gdrive.gdrive()
    
    ###################################
    ###      placer sur la map     ####
    ###################################
    country = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_co', country_code))
    Map.addLayer(country, {}, 'country_code')
    Map.centerObject(country)
     
    #add dataset 
    dataset = ee.ImageCollection('NASA/MEASURES/GFCC/TC/v3').filter(ee.Filter.date('2010-01-01', '2010-12-31'))
    treeCanopyCover = dataset.select('tree_canopy_cover')
    treeCanopyCoverVis = {
        'min': 0.0,
        'max': 100.0,
        'palette': ['ffffff', 'afce56', '5f9c00', '0e6a00', '003800'],
    }
    country_gfcc_2010 = treeCanopyCover.mean().clip(country)
    Map.addLayer(country_gfcc_2010, treeCanopyCoverVis, 'Tree Canopy Cover');

    #load the ecozones 
    gez_2010 = ee.Image('users/bornToBeAlive/gez_2010_wgs84')
    country_gez_2010 =  gez_2010.select('b1').clip(country)
    vizParam = {
        'min': 0, 
        'max': 21,
        'bands': ['b1'],
        'palette': ['black', 'green', 'blue'] 
    }
    Map.addLayer(country_gez_2010, vizParam, 'gez_2010_raster')

    #show the 0 values
    cdl = country_gez_2010.select('b1')
    masked = ee.Image(cdl).updateMask(cdl.eq(0))
    Map.addLayer(masked, {'palette': 'red'}, 'masked_0')
    
    #mask these values from the results
    country_gez_2010 = ee.Image(cdl).updateMask(cdl.neq(0))
    
    #Get the list of values from the mosaic image
    freqHist = country_gez_2010.reduceRegions(**{
        'reducer': ee.Reducer.frequencyHistogram(), 
        'collection': country, 
        'scale': 100, 
    })

    #rewrite the dictionary
    values = ee.List(freqHist.map(getVal).aggregate_array('vals')).flatten().distinct()

    #Run reduceToVectors per class by masking all other classes.
    classes = country_gez_2010.reduceToVectors(**{
        'reducer': ee.Reducer.countEvery(), 
        'geometry': country, 
        'scale': 100,
        'maxPixels': 1e10
    })
    country_gez_2010_vector = ee.FeatureCollection(classes);

    #add the borders on the map
    empty = ee.Image().byte()
    outline = empty.paint(**{
        'featureCollection': country_gez_2010_vector,
        'color': 1,
        'width': 3
    })
    Map.addLayer(outline, {'palette': '000000'}, 'gez_2010_vector')
    
    ########################################
    ##          export useful maps        ##
    ########################################
    su.displayIO(output, 'download treecover')
    
    #create the result folder 
    resultDir = os.path.join(os.path.expanduser('~'), 'zonal_results', country_code, '')
    os.makedirs(resultDir, exist_ok=True)
    
    #export treecover 
    filename = country_code + '_treecover'
    gfcc = resultDir + filename + '.tif'
    if not os.path.isfile(gfcc):
        
        files = drive_handler.get_files(filename)
        if files == []:
            su.displayIO(output, 'import treecover to sepal')
        
            task_config = {
                'image':country_gfcc_2010,
                'description':filename,
                'scale': 30,
                'region':country.geometry(),
                'maxPixels': 1e13
            }
    
            task = ee.batch.Export.image.toDrive(**task_config)
            task.start()
        
            gee.wait_for_completion(filename, output)
            su.displayIO(output, 'image exported to gdrive')
    
        #import tiles to sepal
        su.displayIO(output, 'import tiles to sepal')
    
        files = drive_handler.get_files(filename)
        drive_handler.download_files(files, resultDir)
    
        #find the tiles
        pathname = filename + "*.tif"
        files = [file for file in glob.glob(resultDir + pathname)]
        
        #run the merge process
        gfcc_tmp = resultDir + filename + '_tmp.tif'
        sgdal.merge(files, out_filename=gfcc_tmp, v=True, output=output)
        [os.remove(file) for file in files]
        
    
        #compress it 
        gdal.Translate(
            gfcc, 
            gfcc_tmp, 
            creationOptions=['COMPRESS=LZW'],
            outputType=gdalconst.GDT_Byte
        )
        os.remove(gfcc_tmp)
    
    #export the zones 
    filename = country_code + '_zones'
    zone = resultDir + filename + '.tif'
    if not os.path.isfile(zone):
        
        
        files = drive_handler.get_files(filename)
        if files == []:
            
            su.displayIO(output, 'import ecozones to sepal')
    
            task_config = {
                'image':country_gez_2010,
                'description':filename,
                'scale': 30,
                'region':country.geometry(),
                'maxPixels': 1e13
            }
    
            task = ee.batch.Export.image.toDrive(**task_config)
            task.start()
        
            gee.wait_for_completion(filename, output)
            su.displayIO(output, 'image exported to gdrive')
            
            
    
        #import tiles to sepal 
        su.displayIO(output, 'import tiles to sepal')
    
        files = drive_handler.get_files(filename)
        drive_handler.download_files(files, resultDir)
    
        #find the tiles
        pathname = filename + "*.tif"
        files = [file for file in glob.glob(resultDir + pathname)]
        
        #run the merge process
        zone_tmp = resultDir + filename + '_tmp.tif'
        sgdal.merge(files, out_filename=zone_tmp, v=True, output=output)
        [os.remove(file) for file in files]
    
        #compress it 
        gdal.Translate(
            zone, 
            zone_tmp, 
            creationOptions=['COMPRESS=LZW'], 
            outputType=gdalconst.GDT_Byte
        )
        os.remove(zone_tmp)
    
    ########################################
    ##      run zonal analysis            ##
    ########################################
    raw_data = resultDir + country_code + '_raw_data.txt'
    if not os.path.isfile(raw_data):
        
        su.displayIO(output, 'Compute zonal analysis')
        
        command = [
            'oft-zonal_large_list.py',
            '-i', gfcc,
            '-um', zone,
            '-o', raw_data
        ]
    
        print(' '.join(command))
    
        kwargs = {
            'args' : command,
            'cwd' : os.path.expanduser('~'),
            'stdout' : subprocess.PIPE,
            'stderr' : subprocess.PIPE,
            'universal_newlines' : True
        }
    
        with subprocess.Popen(**kwargs) as p:
            for line in p.stdout:
                su.displayIO(output, line)
                

    #######################################
    ###  lire et concatener les données ###
    #######################################
    df = pd.read_csv(raw_data, sep=' ', header=None)
    names = ['code', 'total'] + ['tc{:02}'.format(i) for i in range (df.shape[1]-2)]
    df.columns = names

    #ajouter les colonnes maquantes 
    for i in range(df.shape[1]-2, 100):
        df['tc{:02}'.format(i)] = [0 for j in range(len(df))]
    
    #add zones
    ecozones =[get_ecozones()[code] for code in df['code']]
    df['zone'] = ecozones
    
    #############################
    ####   exporter     #########
    #############################
    out_stats = os.path.join(resultDir, country_code + '_stats.csv') 
    df.to_csv(out_stats, index=False) 
    
    #############################
    ##    tracer les figures ####
    #############################
    
    list_zones = get_ecozones()

    x_sc = bq.OrdinalScale()
    ax_x = bq.Axis(label='treecover', scale=x_sc)

    figs = []
    for ecozone in ecozones:
        tmp = df.loc[df['zone'] == ecozone]
        values = tmp.filter(items=['tc{:02}'.format(i) for i in range(100)])
        
        y_sc = bq.LinearScale(max=float(values.max(axis=1).array[0]))
        ax_y = bq.Axis(label='surface (px)', scale=y_sc, orientation='vertical')
        
        x= [i for i in range(100)]
        y = [values['tc{:02}'.format(i)].array[0] for i in range(100)]
    
        mark = bq.Bars(x=x, y=y, scales={'x': x_sc, 'y': y_sc})
    
        fig_hist = bq.Figure(
            title=ecozone,
            marks=[mark], 
            axes=[ax_x, ax_y], 
            padding_x=0.025, 
            padding_y=0.025
        )
    
        fig_hist.layout.width = 'auto'
        fig_hist.layout.height = 'auto'
        fig_hist.layout.min_width = "300px"
        fig_hist.layout.min_height = '300px'
    
        figs.append(fig_hist)

    
    return figs, out_stats


def get_ecozones():
    #create the list of zones
    #as ther are no names in the tiff file 
    list_zones = {
        41: 'Boreal coniferous forest',
        43: 'Boreal mountain system',
        42: 'Boreal tundra woodland',
        50: 'Polar',
        24: 'Subtropical desert',
        22: 'Subtropical dry forest',
        21: 'Subtropical humid forest',
        25: 'Subtropical mountain system',
        23: 'Subtropical steppe',
        32: 'Temperate continental forest',
        34: 'Temperate desert',
        35: 'Temperate mountain system',
        31: 'Temperate oceanic forest',
        33: 'Temperate steppe',
        15: 'Tropical desert',
        13: 'Tropical dry forest',
        12: 'Tropical moist forest',
        16: 'Tropical mountain system',
        11: 'Tropical rainforest',
        14: 'Tropical shrubland',
        90: 'Water'
    }
    
    return list_zones

