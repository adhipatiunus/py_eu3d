#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 20:48:44 2022

@author: adhipatiunus
"""

x_min, x_max, y_min, y_max = 0, 1, 0, 1

def partition(x_min, x_max, y_min, y_max, d_min):
    if x_max - x_min <= d_min or y_max - y_min <= d_min:
        return x_min, x_max, y_min, y_max
    x_min, x_max, y_min, y_max = 
    