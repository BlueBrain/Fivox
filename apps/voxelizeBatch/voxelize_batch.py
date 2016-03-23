#!/usr/bin/env python
"""
Usage: voxelize_batch.py --config file.config

Launch 'voxelize' in batch mode using sbatch to generate volumes on cluster
nodes. Based on the 'livre_batch' tool in BlueBrain/Livre
"""

import argparse
import glob
import json
import math
import os
import subprocess

__author__ = "Jafet Villafranca"
__email__ = "jafet.villafrancadiaz@epfl.ch"
__copyright__ = "Copyright 2016, EPFL/Blue Brain Project"

# pylint: disable=W0142

SECTION_SLURM = 'slurm'
SLURM_NAME = 'job_name'
SLURM_TIME = 'job_time'
SLURM_QUEUE = 'queue'
SLURM_ACCOUNT = 'account'
SLURM_OUTPUTDIR = 'output_dir'
SLURM_NODES = 'nodes'
SLURM_TASKS_PER_NODE = 'tasks_per_node'

SECTION_VOXELIZE = 'voxelize'
VOXELIZE_STARTFRAME = 'start_frame'
VOXELIZE_ENDFRAME = 'end_frame'
VOXELIZE_MAXFRAMES = 'max_frames'
VOXELIZE_VOLUME = 'volume'

EXAMPLE_JSON = 'example.json'


def find_voxelize():
    """
    Search for voxelize executable in PATH and return result
    """

    from distutils import spawn
    voxelize_path = spawn.find_executable("voxelize")
    if not voxelize_path:
        print("Cannot find voxelize executable in PATH")
        return False
    print("Using voxelize executable '{0}'".format(voxelize_path))
    return True

class VoxelizeBatch(object):
    """
    Submits sbatch jobs to generate volume files using the voxelize app in find_voxelize
    by using a configuration file for setup.
    """

    def __init__(self, verbose, dry_run):
        self.verbose = verbose
        self.dry_run = dry_run
        self.dict = {}
        self.default_dict = {}
        self._fill_default_dict()

    def _fill_default_dict(self):
        """
        Setup default values for all supported options in the configuration file
        """

        self.default_dict = {
            SECTION_SLURM: {
                SLURM_NAME: 'voxelize_batch',
                SLURM_TIME: '06:00:00',
                SLURM_QUEUE: 'prod',
                SLURM_ACCOUNT: 'proj3',
                SLURM_OUTPUTDIR: '.',
                SLURM_NODES: 1,
                SLURM_TASKS_PER_NODE: 16},
            SECTION_VOXELIZE: {
                VOXELIZE_STARTFRAME: 0,
                VOXELIZE_ENDFRAME: 100,
                VOXELIZE_MAXFRAMES: 20,
                VOXELIZE_VOLUME: ''}}

    def _build_sbatch_script(self, start, end):
        """
        Build sbatch script for a certain frame range
        """

        values = self.dict
        values['start'] = start
        values['end'] = end

        sbatch_script = '\n'.join((
            "#!/bin/bash",
            "#SBATCH --job-name=\"{slurm[job_name]}\"",
            "#SBATCH --time={slurm[job_time]}",
            "#SBATCH --partition={slurm[queue]}",
            "#SBATCH --account={slurm[account]}",
            "#SBATCH --nodes={slurm[nodes]}",
            "#SBATCH --ntasks-per-node={slurm[tasks_per_node]}",
            "#SBATCH --output={slurm[output_dir]}/%j_out.txt",
            "#SBATCH --error={slurm[output_dir]}/%j_err.txt",
            "",
            "voxelize --volume {voxelize[volume]} --frames \"{start} {end}\" "\
            "--output {volume}"
        )).format(**values)

        if self.verbose:
            print(sbatch_script)
        return sbatch_script

    def write_example_config(self):
        """
        Write example configuration to current directory
        """

        with open(EXAMPLE_JSON, 'w') as configfile:
            json.dump(self.default_dict, configfile, sort_keys=True, indent=4,
                      ensure_ascii=False)
        print("Wrote {0} to current directory".format(EXAMPLE_JSON))

    def read_config(self, config):
        """
        Read configuration file and validate content
        """

        with open(config) as configfile:
            self.dict = json.loads(configfile.read())

        volume = self.dict.get(SECTION_VOXELIZE).get(VOXELIZE_VOLUME, '')
        if not volume:
            print("Error: Need valid volume URI")
            return False

        self.dict['volume'] = "{slurm[output_dir]}/{slurm[job_name]}_".format(**self.dict)

        return True

    def submit_jobs(self):
        """
        Submit jobs from frame range specified in configuration, but checks
        for existing volumes in output directory to submit jobs only for
        missing volumes.
        """

        voxelize_dict = self.dict[SECTION_VOXELIZE]
        start_frame = voxelize_dict[VOXELIZE_STARTFRAME]
        end_frame = voxelize_dict[VOXELIZE_ENDFRAME]

        outdir = self.dict[SECTION_SLURM][SLURM_OUTPUTDIR]
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # look for already generated volumes
        vol_prefix = self.dict['volume']
        files = glob.glob('{0}*.mhd'.format(vol_prefix))
        found_volumes = set(int(x[len(vol_prefix):-4]) for x in files)

        if not found_volumes:
            ranges = [(start_frame, end_frame)]
        else:
            # find missing frames
            ideal_range = set(range(start_frame, end_frame))
            missing_volumes = list(ideal_range - found_volumes)
            missing_volumes.sort()

            if not missing_volumes:
                print("No missing volumes found, no jobs will be submitted.")
                return
            print("Found {0} missing volumes".format(len(missing_volumes)))

            def _calc_ranges(i):
                """
                http://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
                """
                from itertools import groupby
                for a, b in groupby(enumerate(i), lambda xy: (xy[1] - xy[0])):
                    b = list(b)
                    yield b[0][1], b[-1][1]
            ranges = _calc_ranges(missing_volumes)

        # Submit job(s) for range(s)
        idx = 1
        for sub_range in ranges:
            idx += self._submit_jobs_for_range(idx, sub_range[0],
                                               sub_range[1] + 1)

        if self.dry_run:
            print("{0} job(s) not submitted (dry run)\n".format(idx-1))
        else:
            print("{0} job(s) submitted, find outputs in {1}\n".format(idx-1,
                                                                       outdir))
        return


    def _submit_jobs_for_range(self, idx, start_frame, end_frame):
        """
        Submit batch jobs for a range of frames. Does rebalancing of maximum
        frames per job, according to max frames from configuration and given
        frame range.
        """

        voxelize_dict = self.dict[SECTION_VOXELIZE]
        batch_size = voxelize_dict[VOXELIZE_MAXFRAMES]
        num_frames = end_frame - start_frame

        num_jobs = int(math.ceil(float(num_frames) / float(batch_size)))
        batch_size = int(math.ceil(float(num_frames) / float(num_jobs)))

        print("Create {0} job(s) with {1} frame(s) each".format(num_jobs,
                                                                batch_size))

        for batch_start in range(start_frame, end_frame, batch_size):
            start = batch_start
            end = min(batch_start + batch_size, end_frame)

            sbatch_script = self._build_sbatch_script(start, end)
            print("Submit job {0} for frames {1} to {2}...".format(idx, start,
                                                                   end))
            idx += 1
            if not self.dry_run:
                sbatch = subprocess.Popen(['sbatch'], stdin=subprocess.PIPE)
                sbatch.communicate(input=sbatch_script)

        return num_jobs


def main():
    """
    Entry point for voxelize batch application does argument parsing and
    calls voxelize_batch class accordingly.
    """

    parser = argparse.ArgumentParser(description="Submit sbatch job(s) \
                                     launching voxelize to generate volumes")
    parser.add_argument("-c", "--config", help="path to config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse config file, but do not submit any jobs")
    parser.add_argument("-e", "--example-config", action="store_true",
                        help="write example.json to current directory")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print more information")
    args = parser.parse_args()

    voxelize_batch = VoxelizeBatch(args.verbose, args.dry_run)

    if args.example_config:
        voxelize_batch.write_example_config()
        exit()

    if not args.config:
        parser.print_help()
        exit()

    if not voxelize_batch.read_config(args.config):
        exit()

    if not find_voxelize():
        exit()

    voxelize_batch.submit_jobs()

if __name__ == "__main__":
    main()
