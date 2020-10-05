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
from sepal_ui import mapping as sm
import time
from pathlib import Path
import ipyvuetify as v
from sepal_ui import sepalwidgets as sw

ee.Initialize()

def getVal(feat):
    feat = ee.Feature(feat)
    vals = ee.Dictionary(feat.get('histogram')).keys()
    return ee.Feature(None, {'vals': vals})

def run_zonal_computation(assetId, output):
    
    list_zones = get_ecozones()
    
    #get the aoi name 
    aoi_name = Path(assetId).stem.replace('aoi_', '')
    
    #create the result folder 
    resultDir = os.path.join(os.path.expanduser('~'), 'zonal_results', aoi_name, '')
    os.makedirs(resultDir, exist_ok=True)
    
    #create the map
    Map = sm.SepalMap(['CartoDB.Positron'])
    Map.add_legend(legend_keys=list(get_ecozones().values()), legend_colors=list(get_colors().values()), position='topleft')
    
    ###################################
    ###      placer sur la map     ####
    ###################################
    output.add_live_msg('viszualiser data')
    
    aoi = ee.FeatureCollection(assetId)
    Map.addLayer(aoi, {}, 'aoi')
    Map.zoom_ee_object(aoi.geometry())
     
    #add dataset 
    dataset = ee.ImageCollection('NASA/MEASURES/GFCC/TC/v3').filter(ee.Filter.date('2010-01-01', '2010-12-31'))
    treeCanopyCover = dataset.select('tree_canopy_cover')
    treeCanopyCoverVis = {
        'min': 0.0,
        'max': 100.0,
        'palette': ['ffffff', 'afce56', '5f9c00', '0e6a00', '003800'],
    }
    country_gfcc_2010 = treeCanopyCover.mean().clip(aoi)
    Map.addLayer(country_gfcc_2010, treeCanopyCoverVis, 'Tree Canopy Cover');

    #load the ecozones 
    gez_2010 = ee.Image('users/bornToBeAlive/gez_2010_wgs84')
    country_gez_2010 =  gez_2010.select('b1').clip(aoi)
    
    Map.addLayer(country_gez_2010.sldStyle(getSldStyle()), {}, 'gez 2010 raster')

    #show the 0 values
    cdl = country_gez_2010.select('b1')
    masked = ee.Image(cdl).updateMask(cdl.eq(0))
    Map.addLayer(masked, {'palette': 'red'}, 'masked 0 data')
    
    #mask these values from the results
    country_gez_2010 = ee.Image(cdl).updateMask(cdl.neq(0))
    
    #Get the list of values from the mosaic image
    freqHist = country_gez_2010.reduceRegions(**{
        'reducer': ee.Reducer.frequencyHistogram(), 
        'collection': aoi, 
        'scale': 100, 
    })

    #rewrite the dictionary
    values = ee.List(freqHist.map(getVal).aggregate_array('vals')).flatten().distinct()

    #Run reduceToVectors per class by masking all other classes.
    classes = country_gez_2010.reduceToVectors(**{
        'reducer': ee.Reducer.countEvery(), 
        'geometry': aoi, 
        'scale': 100,
        'maxPixels': 1e10
    })
    country_gez_2010_vector = ee.FeatureCollection(classes);

    #add the borders on the map
    empty = ee.Image().byte()
    outline = empty.paint(**{
        'featureCollection': country_gez_2010_vector,
        'color': 1,
        'width': 1
    })
    Map.addLayer(outline, {'palette': '000000'}, 'gez 2010 borders')
    

    #export raw data
    out_dir = os.path.join(os.path.expanduser('~'), 'downloads')
    raw_stats = os.path.join(resultDir, aoi_name + '_raw.csv') 
    
    if not os.path.isfile(raw_stats):
        output.add_live_msg('compute the zonal analysis')
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
            output.add_live_msg(f.getvalue())
        
            #check if the computation have finished on a file 
            if not os.path.isfile(raw_stats):
                #cannot use tile_scale > 16 
                if cpt == 3:
                    raise Exception("Aoi is to big")
                
                #change the tile_scale and relaunch the process
                output.add_live_msg(f.getvalue(), 'error')
                time.sleep(2)
                cpt += 1
                output.add_live_msg('Increasing tile_scale ({})'.format(2**cpt), 'warning')
                time.sleep(2)
            else:
                computation = True

    #######################################
    ###  lire et concatener les données ###
    #######################################
    
    out_stats = os.path.join(resultDir, aoi_name + '_stats.csv') 
    
    if not os.path.isfile(out_stats):
        output.add_live_msg('read and concatenate the data')
        data = pd.read_csv(raw_stats, sep=',')

        output.add_live_msg('remove no_data')
        #enlever les lignes Nan 
        #(vérifier dans le tableau d'avant qu'elles ne sont pas trop grandes)
        data = data.dropna()
        
        #recuperer les valeurs de label
        ecozones = data.label.unique()

        output.add_live_msg('merge')
        #aggreger les lignes avec les même valeurs 
        dummy = []
        for i in range(100):
            dummy.append("{:.1f}".format(i))
        stats = pd.DataFrame(dummy, columns=['treecover'])

        output.add_live_msg('labeliser')
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
    ##    create the layout    ##
    #############################
    output.add_live_msg('create the layout')
    
    stats = pd.read_csv(out_stats)
    
    #recuperer les noms de label
    ecozones = stats.columns[1:]

    x_sc = bq.LinearScale()
    ax_x = bq.Axis(label='treecover', scale=x_sc)#, tick_values=np.array([0,10,20,30]))
    
    x= []
    for i in range(100):
        x.append(i)
    
    figs = []
    for ecozone in ecozones:
        
        #get the # of the ecozone 
        for index, value in get_ecozones().items():
            if value == ecozone: 
                zone_index = index
                break
            
        y_sc = bq.LinearScale(max=stats[ecozone].max())
        ax_y = bq.Axis(label='surface (px)', scale=y_sc, orientation='vertical', tick_format='.2e')
        y = []
        for index, row in stats.iterrows():
            y.append(row[ecozone])
    
        mark = bq.Bars(x=x, y=y, scales={'x': x_sc, 'y': y_sc}, colors=[get_colors()[zone_index]], stroke='black', padding=0.1)
    
        fig_hist = bq.Figure(
            title=ecozone,
            marks=[mark], 
            axes=[ax_x, ax_y]
        )
    
        figs.append(fig_hist)
        
    
    #create the partial layout 
    children = [
        v.Flex(xs12=True, class_='pa-0', children=[sw.DownloadBtn('Download .csv', path=out_stats)]),
        v.Flex(xs12=True, class_='pa-0', children=[Map]),
        v.Flex(xs12=True, class_='pa-0', children=figs)
    ]
    
    output.add_live_msg('complete', 'success')
    
    return children

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
            AUTHORITY["ESRI","54009"]]'
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

def get_colors():
    #create the color for each zones
    #as ther are no names in the tiff file 
    zones_colors = {
        41: "#0266F2",
        43: "#C7ECFF",
        42: "#6EDCFF",
        50: "#BCD2FF",
        24: "#FEF3CD",
        22: "#974000",
        21: "#3F0D01",
        25: "#EC9664",
        23: "#FDE499",
        32: "#1FE62C",
        34: "#F1FFCD",
        35: "#85FF41",
        31: "#008C01",
        33: "#D9FF97",
        15: "#FEE6D9",
        13: "#E31A32",
        12: "#BD014D",
        16: "#FE8C4D",
        11: "#7F004E",
        14: "#FECCB3",
        90: "#FEFFFF"
    }
    
    return zones_colors

def getSldStyle():
    #Define an SLD style of discrete intervals to apply to the image.
    color_map_entry = '\n<ColorMapEntry color="{0}" quantity="{1}" label="{2}"/>' 
    
    # TODO can do it programatically
    sld_intervals = '<RasterSymbolizer>' 
    sld_intervals += '\n<ColorMap type="intervals" extended="false" >' 
    
    list_zones = get_ecozones()
    zones_colors = get_colors()
    
    for zoneId in list_zones:
        sld_intervals += color_map_entry.format(zones_colors[zoneId], zoneId, list_zones[zoneId])
                                            
    sld_intervals += '\n</ColorMap>'
    sld_intervals += '\n</RasterSymbolizer>'    
    
    return sld_intervals