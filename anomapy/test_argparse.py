#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 15:06:12 2020

@author: ben
"""

import argparse 



parser = argparse.ArgumentParser()
parser.add_argument("-a")
args = parser.parse_args()

print(args)
