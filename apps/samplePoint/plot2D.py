#!/usr/bin/env python
"""
Usage: plot2D.py --input data.txt [--output graph.png]

Tool to plot as a 2D graph the evolution of the data at a specific point over
time
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Jafet Villafranca"
__email__ = "jafet.villafrancadiaz@epfl.ch"
__copyright__ = "Copyright 2016, EPFL/Blue Brain Project"


def main():

    parser = argparse.ArgumentParser(description="Tool to plot as a 2D graph "
                                     "the evolution of data over time, loading "
                                     "the values from file")
    parser.add_argument("-i", "--input", help="input file containing "
                        "all the values to plot: one line per value, using "
                        "the 'timestamp value' format")
    parser.add_argument("-o", "--output", help="output file name to save the "
                        "resulting 2D graph as an image (SVG format). If not "
                        "specified, open a window showing an interactive plot")
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        exit()

    x = []
    y = []
    nline = 0
    for line in open(args.input):
        nline += 1
        l = line.strip()
        if l and not l.startswith("#"):
            values = line.split()
            if len(values) != 2:
                print("Skipping line " + str(nline) + ". Please use a "
                      "[timestamp value] pair per line.")
                continue
            try:
                xvalue = float(values[0])
                yvalue = float(values[1])
            except ValueError:
                print("Skipping line " + str(nline) + ". Please use numeric "
                      "values.")
                continue
            x.append(xvalue)
            y.append(yvalue)

    plt.plot(x, y)

    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")

    if args.output:
        imagefile = args.output
        if not imagefile.strip().endswith(".svg"):
            imagefile += ".svg"
        plt.savefig(imagefile, format="svg")
        print("2D graph saved as " + imagefile)
    else:
        plt.show()

if __name__ == "__main__":
    main()
