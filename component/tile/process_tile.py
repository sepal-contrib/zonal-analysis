import ipyvuetify as v

import sepal_ui.sepalwidgets as sw
from sepal_ui.scripts import utils as su
from component.scripts import zonal_computation as zc
from component.message import ms

class ProcessTile(v.Card):
    
    def __init__(self, aoi, results_tile, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.aoi = aoi
        self.results_tile = results_tile
        
        self.alert = sw.Alert()
        self.btn = sw.Btn(ms.buttons.process.label)
        
        self.children = [
            self.alert,
            self.btn
        ]
        
        self.btn.on_event('click', self.process_start)
    
    @su.loading_button(debug=True)
    def process_start(self, widget,*args):

        #check that the asset is defined
        if not self.aoi.view.model.feature_collection:
            raise Exception(ms.process.no_aoi)

        layout = zc.run_zonal_computation(self.aoi, self.alert)
        self.results_tile.set_content(layout)

        return 
