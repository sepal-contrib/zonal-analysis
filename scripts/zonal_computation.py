import ee
import geemap
import os
import numpy as np
import csv
import pandas as pd
import bqplot as bq
import json
import io
from contextlib import redirect_stdout
from sepal_ui.scripts import utils as su
import time

ee.Initialize()

def getVal(feat):
    feat = ee.Feature(feat)
    vals = ee.Dictionary(feat.get('histogram')).keys()
    return ee.Feature(None, {'vals': vals})

def run_zonal_computation(country_code, Map, output):
    
    list_zones = get_ecozones()
    
    #create the result folder 
    resultDir = os.path.join(os.path.expanduser('~'), 'zonal_results', country_code, '')
    os.makedirs(resultDir, exist_ok=True)
    
    ###################################
    ###      placer sur la map     ####
    ###################################
    su.displayIO(output, 'viszualiser data')
    
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
    

    #export raw data
    out_dir = os.path.join(os.path.expanduser('~'), 'downloads')
    raw_stats = os.path.join(resultDir, country_code + '_raw.csv') 
    
    if not os.path.isfile(raw_stats):
        print('compute the zonal analysis')
        #compute the zonal analysis
        computation = False
        cpt = 0
        while not computation:
            f = io.StringIO()
            with redirect_stdout(f):
                geemap.zonal_statistics(
                    in_value_raster = country_gfcc_2010, 
                    in_zone_vector  = country_gez_2010_vector, 
                    out_file_path   = raw_stats, 
                    statistics_type = 'FIXED_HIST', 
                    scale           = 100, 
                    crs             = getConformProj(), 
                    hist_min        = 0, 
                    hist_max        = 100, 
                    hist_steps      = 100,
                    tile_scale      = 2**cpt
                )
            su.displayIO(output, f.getvalue())
        
            #check if the computation have finished on a file 
            if not os.path.isfile(raw_stats):
                #cannot use tile_scale > 16 
                if cpt == 3:
                    raise Exception("Aoi is to big")
                
                #change the tile_scale and relaunch the process
                su.displayIO(output, f.getvalue(), 'error')
                time.sleep(2)
                cpt += 1
                su.displayIO(output, 'on augmente le tile_scale ({})'.format(2**cpt), 'warning')
                time.sleep(2)
            else:
                computation = True

    #######################################
    ###  lire et concatener les données ###
    #######################################
    
    out_stats = os.path.join(resultDir, country_code + '_stats.csv') 
    
    if not os.path.isfile(out_stats):
        su.displayIO(output, 'lire et concatener les données')
        data = pd.read_csv(raw_stats, sep=',')

        su.displayIO(output, 'retirer les no_data')
        #enlever les lignes Nan 
        #(vérifier dans le tableau d'avant qu'elles ne sont pas trop grandes)
        data = data.dropna()
        
        #recuperer les valeurs de label
        ecozones = data.label.unique()

        su.displayIO(output, 'merge')
        #aggreger les lignes avec les même valeurs 
        dummy = []
        for i in range(100):
            dummy.append("{:.1f}".format(i))
        stats = pd.DataFrame(dummy, columns=['treecover'])

        su.displayIO(output, 'labeliser')
        for index, ecozone in enumerate(ecozones): 
            patches = data.loc[data.label == ecozone]
            label = []
            for i in range(100):
                label.append(["{:.1f}".format(i), 0])
        
            for index, row in patches.iterrows():
                tmp = json.loads(row['histogram'])        
                for index, value in enumerate(tmp):
                    label[index][1] += value[1]
            
            label = pd.DataFrame(label, columns=['treecover', list_zones[ecozone]])
            stats = pd.merge(left=stats, right=label, left_on='treecover', right_on='treecover')
            
        #exporter 
        stats.to_csv(out_stats, index=False) 
    
    #############################
    ##    tracer les figures ####
    #############################
    su.displayIO(output, 'tracer les figures')
    
    stats = pd.read_csv(out_stats)
    
    #recuperer les noms de label
    ecozones = stats.columns[1:]

    x_sc = bq.OrdinalScale()
    ax_x = bq.Axis(label='treecover', scale=x_sc)
    
    x= []
    for i in range(100):
        x.append(i)
    
    figs = []
    for ecozone in ecozones:
        y_sc = bq.LinearScale(max=stats[ecozone].max())
        ax_y = bq.Axis(label='surface (px)', scale=y_sc, orientation='vertical')
        y = []
        for index, row in stats.iterrows():
            y.append(row[ecozone])
    
        mark = bq.Bars(x=x, y=y, scales={'x': x_sc, 'y': y_sc})
    
        fig_hist = bq.Figure(
            title=ecozone,
            marks=[mark], 
            axes=[ax_x, ax_y]
        )
    
        figs.append(fig_hist)
    
    return figs, out_stats

def getConformProj():
    
    wkt = """
        PROJCS["World_Mollweide",
            GEOGCS["GCS_WGS_1984",
                DATUM["WGS_1984",
                    SPHEROID["WGS_1984",6378137,298.257223563]],
                PRIMEM["Greenwich",0],
                UNIT["Degree",0.017453292519943295]],
            PROJECTION["Mollweide"],
            PARAMETER["False_Easting",0],
            PARAMETER["False_Northing",0],
            PARAMETER["Central_Meridian",0],
            UNIT["Meter",1],
            AUTHORITY["EPSG","54009"]]'
    """

    return ee.Projection(wkt)

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