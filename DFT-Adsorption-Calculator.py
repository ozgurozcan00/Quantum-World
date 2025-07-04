# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:48:49 2025

@author: ozgur.ozcan
"""
energy_sio2 = -1137.562172

energy_zno = -502.8382788106

energy_sio2zno = -1639.500090075

ads_val1 = (energy_sio2zno - (energy_sio2 + energy_zno))*13.6057

print("Adsorption Energy:",ads_val1,"eV")
