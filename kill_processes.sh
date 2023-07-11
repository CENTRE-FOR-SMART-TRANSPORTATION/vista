#!/bin/bash

# Find PID of process controller and send a SIGTERM (tries to kill the process) to it
kill $(ps S | grep 'process_controller_v2.sh' | grep -v grep | awk '{print  $  1 }')

# Find PIDs of each process and also send a SIGTERM (tries to kill the processes) to them
kill $(ps S | grep -E 'bash ./examples/processes/single/process_i.sh' | grep -v grep | awk '{print  $  1 }')

echo "Processes killed."

