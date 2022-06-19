#!/usr/bin/sh
HADOOP2='hadoop fs -Dfs.default.name=hdfs://ss-wxg-3-v2 -Dhadoop.job.ugi=tdw_zhiyuanxu:qq123!@#,wxbiz_offline_datamining '
alias hdfs2='hadoop fs -Dfs.default.name=hdfs://ss-wxg-3-v2 -Dhadoop.job.ugi=tdw_zhiyuanxu:qq123!@#,wxbiz_offline_datamining '
echo $1
echo $2
rm -rf $1
mkdir -p $1
${HADOOP2} -get $2 $1
chmod -R 777 $1