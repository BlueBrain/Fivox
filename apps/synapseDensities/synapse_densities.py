#!/usr/bin/env python
"""
Usage: synapse_densities.py --help

Voxelizes and optionally shows synapse densities.
"""

import argparse
import hashlib
import os
import subprocess

__author__ = "Daniel Nachbaur"
__email__ = "daniel.nachbaur@epfl.ch"
__copyright__ = "Copyright 2016, EPFL/Blue Brain Project"

def find_executable(executable):
    """
    Search for executable in PATH and return result
    """

    from distutils import spawn
    executable_path = spawn.find_executable(executable)
    if not executable_path:
        print("Cannot find {0} executable in PATH".format(executable))
        return False
    return True

class Launcher(object): # pylint: disable=too-few-public-methods
    """
    Launcher for a tool showing synapse density volume.
    """

    def __init__(self, args):
        """
        Setup volume URI for livre and/or voxelize from args.
        """

        self._usevgl = 'VGL_CLIENT' in os.environ
        self._args = args
        self._args.resolution = 1./self._args.resolution
        self._volume = 'fivoxsynapses://{config}?resolution={resolution}&'
        if args.target:
            self._volume += 'target={target}'
            self._outname = 'density_{target}'.format(**vars(self._args))
        else:
            self._volume += 'preTarget={projection[0]}&postTarget={projection[1]}'
            self._outname = 'density_{projection[0]}_{projection[1]}'.format(**vars(self._args))
        if args.fraction:
            self._volume += '&gidFraction={fraction}'
        if args.reference:
            self._volume += '&reference={reference}'
        else:
            self._volume += '&extend=200'
        if args.datarange:
            self._volume += '&inputMin={datarange[0]}&inputMax={datarange[1]}'

        self._volume = self._volume.format(**vars(self._args))
        if args.verbose:
            print(self._volume)

        hash_object = hashlib.md5(str(self._volume))
        self._outname += "_{0}.nrrd".format(hash_object.hexdigest()[:7])
        self._outname = os.path.abspath(self._outname)

    def launch(self):
        """
        Launch livre or voxelize and/or paraview.
        """

        if os.path.exists(self._outname):
            print("Reusing previously generated volume {0}".format(self._outname))

        if self._args.tool == 'livre':
            if os.path.exists(self._outname):
                volume = "raw://{0}".format(self._outname)
            else:
                volume = self._volume

            args = ['livre', '--volume', volume, '--max-lod 0']
            if self._usevgl:
                args.insert(0, 'vglrun')
            subprocess.call(args)

        if self._args.tool == 'voxelize' or self._args.tool == 'paraview':
            if not os.path.exists(self._outname):
                subprocess.call(['voxelize', '--volume', self._volume,
                                 '-o', self._outname])

            if self._args.tool == 'paraview':
                args = ['paraview', self._outname]
                if self._usevgl:
                    args.insert(0, 'vglrun')
                subprocess.call(args)


def main():
    """
    Find the available tools and launch them according to user input.
    """

    tools = ['livre', 'paraview', 'voxelize']
    executables = list()
    for tool in tools:
        if find_executable(tool):
            executables.append(tool)

    if not executables:
        print("Cannot find any tools in PATH")
        exit()

    parser = argparse.ArgumentParser(
        description="A tool for synapse densities visualization",
        epilog="If paraview or voxelize was launched, it creates a "
               "'density_{target}_{hash}.nrrd' volume file in the current "
               "directory which will be reused if this tool was called again "
               "with the same parameters.")
    parser.add_argument("tool", choices=executables,
                        help="Tool to use for volume generation/visualization")
    parser.add_argument("-c", "--config", metavar='<BlueConfig>',
                        help="path to blue config file", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--target", metavar='<target>',
                       help="target for afferent synapses")
    group.add_argument('-p', "--projection", nargs=2, metavar=('pre', 'post'),
                       help="targets for synaptic projections")
    parser.add_argument("-f", "--fraction", metavar='<GID fraction>',
                        help="fraction of GIDs to use [0,1]")
    parser.add_argument("-d", "--datarange", nargs=2, metavar=('min', 'max'),
                        help="data range to use")
    parser.add_argument("-r", "--resolution", metavar="<micrometer/voxel>", default=16, type=float,
                        help="resolution in micrometer/voxel, default 16")
    parser.add_argument("--reference", metavar='<path to volume>', default="",
                        help="path to reference volume for size and resolution setup")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print volume URI")
    args = parser.parse_args()

    launcher = Launcher(args)
    launcher.launch()

if __name__ == "__main__":
    main()
