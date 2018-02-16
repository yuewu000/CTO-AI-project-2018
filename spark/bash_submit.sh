#Path to the BigDL library :
BIGDL_JAR="/home/araji/workarea/BigDL/dist/lib/bigdl-0.5.0-SNAPSHOT-jar-with-dependencies.jar"
BIGDL_PY="/home/araji/workarea/BigDL/dist/lib/bigdl-0.5.0-SNAPSHOT-python-api.zip"
hdfs path where the sequence files have been uploaded 
SEQ_PATH=hdfs:///tmp/arseq
#spark 
CORE_PER_EXECUTOR=75
NUM_EXECUTORS=6
IMG_PER_CORE=4
EXEC_MEMORY=500g
BATCH_SIZE="$(( ${CORE_PER_EXECUTOR} * ${NUM_EXECUTORS} * ${IMG_PER_CORE} ))"
#ML hyperparams
learningRate=0.1
WeightDecay=0.00004
CheckpointIteration=20
#starting the training :
#Submit the job :
        spark2-submit \
       --master yarn --deploy-mode client \
       --executor-cores ${CORE_PER_EXECUTOR} --num-executors ${NUM_EXECUTORS} --executor-memory ${EXEC_MEMORY}  --driver-memory 256g \
       --jars ${BIGDL_JAR} --conf spark.dynamicAllocation.enabled=false  --conf spark.serializer=org.apache.spark.serializer.JavaSerializer --files /home/araji/workarea/metrics.properties \
       --conf spark.metrics.conf=metrics.properties \
       --driver-class-path ${BIGDL_JAR} --class com.intel.analytics.bigdl.models.inception.TrainInceptionV1 ${BIGDL_JAR} \
       --batchSize ${BATCH_SIZE} --learningRate ${learningRate} --weightDecay ${WeightDecay} --checkpointIteration ${CheckpointIteration} -f ${SEQ_PATH} --checkpoint /home/araji/workarea/models
