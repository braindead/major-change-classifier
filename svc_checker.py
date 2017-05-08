#!/usr/bin/python

import sys
import SVC
import argparse
import json

parser = argparse.ArgumentParser(description='Classify changes as major or minor')
parser.add_argument('json_file', type=str, help='path to json file')
parser.add_argument('--debug', dest='debug', action='store_true', default=False, help='print debug information')
parser.add_argument('--extended-fillers', dest='extended_fillers', action='store_true', default=False, help='Use extended fillers filter')
args = parser.parse_args()

if __name__ == '__main__': 
    checker = SVC.Checker()

    with open(args.json_file) as data_file:    
        data = json.load(data_file)

    predictions = checker.predict(data, args.extended_fillers, args.debug)
    if args.debug is False:
        print(json.dumps(predictions))
