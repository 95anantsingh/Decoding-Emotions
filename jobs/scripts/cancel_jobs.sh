#!/bin/bash

scancel $(sacct -n -X --format jobid --name Job1)
scancel $(sacct -n -X --format jobid --name Job2)
scancel $(sacct -n -X --format jobid --name Job3)
scancel $(sacct -n -X --format jobid --name Job4)
scancel $(sacct -n -X --format jobid --name Job5)
scancel $(sacct -n -X --format jobid --name Job6)
scancel $(sacct -n -X --format jobid --name Job7)
scancel $(sacct -n -X --format jobid --name Job8)
