#!/bin/bash

scancel $(sacct -n -X --format jobid --name Job218)
scancel $(sacct -n -X --format jobid --name Job219)
scancel $(sacct -n -X --format jobid --name Job220)
