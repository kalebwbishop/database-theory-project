#!/bin/bash
#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# 
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------
set -e

if [ "$(basename $PWD)" != "perftest" ];
then
  echo "Please execute scripts from directory 'perftest'"
  exit 1;
fi

CMD=$5
BASE=$4

echo "running stratstats"
tstart=$(date +%s.%N)

#${CMD} -f ../algorithms/stratstats.dml \
${CMD} -f ./scripts/stratstats.dml \
  --config conf/SystemDS-config.xml \
  --stats \
  --nvargs X=$1 Xcid=$2 Ycid=$3 O=${BASE}/STATS/s fmt=csv

ttrain=$(echo "$(date +%s.%N) - $tstart - .4" | bc)
echo "StratifiedStatistics on "$1": "$ttrain >> results/times.txt
