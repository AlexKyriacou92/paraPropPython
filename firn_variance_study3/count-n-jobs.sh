#! /usr/bin/bash

squeue -u a969k397 -h -t pending,running -r | wc -l
