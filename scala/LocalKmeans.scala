package com.huawei.clustering

import scala.util.Random

import breeze.linalg.{Vector => BV, DenseVector => BDV, norm => breezeNorm}

import org.apache.spark.Logging

/**
 * An utility object to run K-means locally. This is private to the ML package because it's used
 * in the initialization of KMeans but not meant to be publicly exposed.
 */
object LocalKmeans extends Logging {

  /**
   * Run K-means++ on the weighted point set `points`. This first does the K-means++
   * initialization procedure and then rounds of Lloyd's algorithm.
   */
  def kmeansPlusPlus(
      seed: Int,
      points: Array[BreezeVectorWithNorm],
      weights: Array[Double],
      k: Int,
      maxIterations: Int
  ): Array[BreezeVectorWithNorm] = {
    val rand = new Random(seed)
    val dimensions = points(0).vector.length
    val centers = new Array[BreezeVectorWithNorm](k)

    // Initialize centers by sampling using the k-means++ procedure.
    centers(0) = pickWeighted(rand, points, weights).toDense
    for (i <- 1 until k) {
      // Pick the next center with a probability proportional to cost under current centers
      val curCenters = centers.view.take(i)
      val sum = points.view.zip(weights).map { case (p, w) =>
        w * Kmeans.pointCost(curCenters, p)
      }.sum
      val r = rand.nextDouble() * sum
      var cumulativeScore = 0.0
      var j = 0
      while (j < points.length && cumulativeScore < r) {
        cumulativeScore += weights(j) * Kmeans.pointCost(curCenters, points(j))
        j += 1
      }
      if (j == 0) {
        logWarning("kMeansPlusPlus initialization ran out of distinct points for centers." +
          s" Using duplicate point for center k = $i.")
        centers(i) = points(0).toDense
      } else {
        centers(i) = points(j - 1).toDense
      }
    }

    // Run up to maxIterations iterations of Lloyd's algorithm
    val oldClosest = Array.fill(points.length)(-1)
    var iteration = 0
    var moved = true
    while (moved && iteration < maxIterations) {
      moved = false
      val counts = Array.fill(k)(0.0)
      val sums = Array.fill(k)(
        BDV.zeros[Double](dimensions).asInstanceOf[BV[Double]]
      )
      var i = 0
      while (i < points.length) {
        val p = points(i)
        val index = Kmeans.findClosest(centers, p)._1
        breeze.linalg.axpy(weights(i), p.vector, sums(index))
        counts(index) += weights(i)
        if (index != oldClosest(i)) {
          moved = true
          oldClosest(i) = index
        }
        i += 1
      }
      // Update centers
      var j = 0
      while (j < k) {
        if (counts(j) == 0.0) {
          // Assign center to a random point
          centers(j) = points(rand.nextInt(points.length)).toDense
        } else {
          sums(j) /= counts(j)
          centers(j) = new BreezeVectorWithNorm(sums(j))
        }
        j += 1
      }
      iteration += 1
    }

    if (iteration == maxIterations) {
      logInfo(s"Local KMeans++ reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"Local KMeans++ converged in $iteration iterations.")
    }

    centers
  }

  private def pickWeighted[T](rand: Random, data: Array[T], weights: Array[Double]): T = {
    val r = rand.nextDouble() * weights.sum
    var i = 0
    var curWeight = 0.0
    while (i < data.length && curWeight < r) {
      curWeight += weights(i)
      i += 1
    }
    data(i - 1)
  }
}
