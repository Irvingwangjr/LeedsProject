#!/usr/bin/sh
HADOOP2='hadoop fs -Dfs.default.name=hdfs://ss-wxg-3-v2 -Dhadoop.job.ugi=tdw_evanxcwang:1422333,wxbiz_offline_datamining '
alias hdfs2='hadoop fs -Dfs.default.name=hdfs://ss-wxg-3-v2 -Dhadoop.job.ugi=tdw_evanxcwang:1422333,wxbiz_offline_datamining '
echo $1
echo $2
#${HADOOP2} -rmr -skipTrash $2
if [ -d "$1" ]; then
  ${HADOOP2} -mkdir -p $2
fi
#${HADOOP2} -mkdir $2
${HADOOP2} -put $1 $2
${HADOOP2} -chomd -R 777 $2