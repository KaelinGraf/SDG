#SDG Isaac Sim Dashboard
#Allows users to control and monitor the data generation process while running isaacsim in headless mode, also 
#functions as a visualizer for debugging purposes. 
#The dashboard manages the isaacsim instance from a seperate thread, allowing for automatic restarts/diagnostics if the sim crashes.
#The dashboard also serves as a way for the user to input parameters for the data generation process without needing to edit config files and restart the sim.
#This includes setting paths for saving data, selecting scene types, and setting parameters for the scene
#Kaelin Graf-Ogilvie 2026

import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import QAction
import subprocess
import time
import os








class Dashboard(QMainWindow):
    def __init__(self):
        """
        general init method, starts app and relevant monitoring threads
        """
        super().__init__()
        self._init_ui()
        
        
    
    def _init_ui(self):
        """
        Initialises the UI to it's base state
        """
        self.resize(1920,1080)
        self.setWindowTitle("ISCAR SDG Dashboard")
        
        
        self.label = QLabel()
        self.input = QLineEdit()
        self.input.textChanged.connect(self.label.setText)
        self.input.enterEvent
        
        layout = QVBoxLayout()
        layout.addWidget(self.input)
        layout.addWidget(self.label)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
    def contextMenuEvent(self,e):
        context = QMenu(self)
        context.addAction(QAction("test 1,", self))
        context.addAction(QAction("test 2", self))
        context.addAction(QAction("test 3", self))
        context.exec(e.globalPos())
        
    # def _on_sim_start_callback(self):
    #     """
    #     Responsible for starting the primary isaacsim process, dispatching an async thread to wait for sim start confirmation,
    #     then displaying this information to the user.
    #     """
    #     pass
    
    # def _on_select_scene_config_callback(self):
    #     """
    #     Responsible for opening a file browser for the user to select the relevant scene config JSON file
    #     """
    

def main():
    app = QApplication(sys.argv)
    dash = Dashboard()
    dash.show()
    sys.exit(app.exec())
    

if __name__ == "__main__":
    main()
    

    