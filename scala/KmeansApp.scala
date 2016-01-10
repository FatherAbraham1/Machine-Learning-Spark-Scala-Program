package com.huawei.clustering

import org.apache.log4j.{Level, Logger}

import java.io._

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors

object KmeansApp {

    def main(args: Array[String]) { 
        val k = Integer.valueOf(args(0))   
        val seed = Integer.valueOf(args(1))  
        val partitionNum = java.lang.Double.parseDouble(args(2)) 
        val initSteps = 50
        val initMode = args(3)  // "partition" for kmeans++, "random" for normal kmeans, 
                                // "diameter" for diameter kmeans
        val input = args(4)
        val output = args(5) + "result" + 
                (System.nanoTime()/1e9).asInstanceOf[Int]
        val conf = new SparkConf().setAppName("kmeans with Artificial Data")
        val sc = new SparkContext(conf)

        val examples = sc.textFile(input).map { line => 
            Vectors.dense(line.split(' ').map(_.toDouble))
        }.cache()

        val model = new Kmeans()
            .setInitializationMode(initMode)
            .setK(k)
            .setSeed(seed)
            .setMaxIterations(30) //100
            .setPartitionNumber(partitionNum)
            .setInitializationSteps(initSteps)
            .run(examples)

        // write result into disk
        val resultRDD = model.predict(examples)
        resultRDD.saveAsTextFile(output)

        // persist all results and meta data into disk
        val writer = new PrintWriter(new File(output + "/runInfo" + ".txt"))
        if(initMode.equals("patition")) {
            writer.write("The partition number parameter is: " + partitionNum + "\r")
            writer.write("The init steps number parameter is:" + initSteps + "\r")
        }

        for(i <- 0 until k) {
            writer.write("Init center of Cluster " + i + model.initCenters(i).toString + "\r")
            writer.write("Final center of Cluster " + i + model.clusterCenters(i).toString + "\r")
        }

        writer.close()
        sc.stop()
    }
}
