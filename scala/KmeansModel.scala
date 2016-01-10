package com.huawei.clustering

import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vector

/**
 * A clustering model for K-means. Each point belongs to the cluster with the closest center.
 */
 // this part has been modified
class KmeansModel (val clusterCenters: Array[Vector], val initCenters: Array[Vector]) extends Serializable {

  /** Total number of clusters. */
  def k: Int = clusterCenters.length

  /** Returns the cluster index that a given point belongs to. */
  def predict(point: Vector): Int = {
    Kmeans.findClosest(clusterCentersWithNorm, new BreezeVectorWithNorm(point))._1
  }

  /** Maps given points to their cluster indices. */
  def predict(points: RDD[Vector]): RDD[Int] = {
    val centersWithNorm = clusterCentersWithNorm
    val bcCentersWithNorm = points.context.broadcast(centersWithNorm)
    points.map(p => Kmeans.findClosest(bcCentersWithNorm.value, new BreezeVectorWithNorm(p))._1)
  }

  /** Maps given points to their cluster indices. */
  def predict(points: JavaRDD[Vector]): JavaRDD[java.lang.Integer] =
    predict(points.rdd).toJavaRDD().asInstanceOf[JavaRDD[java.lang.Integer]]

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  def computeCost(data: RDD[Vector]): Double = {
    val centersWithNorm = clusterCentersWithNorm
    val bcCentersWithNorm = data.context.broadcast(centersWithNorm)
    data.map(p => Kmeans.pointCost(bcCentersWithNorm.value, new BreezeVectorWithNorm(p))).sum()
  }

  private def clusterCentersWithNorm: Iterable[BreezeVectorWithNorm] =
    clusterCenters.map(new BreezeVectorWithNorm(_))
}
