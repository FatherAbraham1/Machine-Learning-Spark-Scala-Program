package com.huawei.clustering

import java.io._

import scala.collection.mutable.{Stack, ArrayBuffer}
// modified follow breeze: 
import breeze.linalg.{DenseVector => BDV, Vector => BV, norm => breezeNorm, 
    SparseVector => BSV, squaredDistance => breezeSquaredDistance}

import org.apache.spark.annotation.Experimental
import org.apache.spark.Logging
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{Vector, Vectors, DenseVector} // add DenseVector
import org.apache.spark.mllib.util.MLUtils 
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel // add this


/**
 * K-means clustering with support for multiple parallel runs and a k-means++ like initialization
 * mode (the k-means|| algorithm by Bahmani et al). When multiple concurrent runs are requested,
 * they are executed together with joint passes over the data for efficiency.
 *
 * This is an iterative algorithm that will make multiple passes over the data, so any RDDs given
 * to it should be cached by the user.
 */
class Kmeans (
    var k: Int,   // the number of clusters
    var maxIterations: Int,
    var runs: Int,
    var initializationMode: String,
    var initializationSteps: Int,
    var epsilon: Double,
    var seed: Int,                   // add this
    var partitionNumber: Double      // add this
    ) extends Serializable with Logging {

  /**
   * Constructs a Kmeans instance with default parameters: {k: 2, maxIterations: 20, runs: 1,
   * initializationMode: "k-means||", initializationSteps: 5, epsilon: 1e-4}.
   */
  def this() = this(2, 20, 1, Kmeans.PARALLEL, 5, 1e-4, 1, 4)

  /** Set the number of clusters to create (k). Default: 2. */
  def setK(k: Int): this.type = {
    this.k = k
    this
  }

  // add set seed method
  def setSeed(s: Int): this.type = {
    this.seed = s
    this
  }
  
  /** Set maximum number of iterations to run. Default: 20. */
  def setMaxIterations(maxIterations: Int): this.type = {
    this.maxIterations = maxIterations
    this
  }

  /**
   * Set the initialization algorithm. This can be either "random" to choose random points as
   * initial cluster centers, or "k-means||" to use a parallel variant of k-means++
   * (Bahmani et al., Scalable K-Means++, VLDB 2012). Default: k-means||.
   */
   // modified this
  def setInitializationMode(initializationMode: String): this.type = {
    if (initializationMode != Kmeans.RANDOM && initializationMode != Kmeans.PARALLEL 
      && initializationMode != Kmeans.PARTITION && initializationMode != Kmeans.DIAMETER) {
      throw new IllegalArgumentException("Invalid initialization mode: " + initializationMode)
    }
    this.initializationMode = initializationMode
    this
  }

  /**
   * :: Experimental ::
   * Set the number of runs of the algorithm to execute in parallel. We initialize the algorithm
   * this many times with random starting conditions (configured by the initialization mode), then
   * return the best clustering found over any run. Default: 1.
   */
  @Experimental
  def setRuns(runs: Int): this.type = {
    if (runs <= 0) {
      throw new IllegalArgumentException("Number of runs must be positive")
    }
    this.runs = runs
    this
  }

  /**
   * Set the number of steps for the k-means|| initialization mode. This is an advanced
   * setting -- the default of 5 is almost always enough. Default: 5.
   */
  def setInitializationSteps(initializationSteps: Int): this.type = {
    if (initializationSteps <= 0) {
      throw new IllegalArgumentException("Number of initialization steps must be positive")
    }
    this.initializationSteps = initializationSteps
    this
  }

  /**
   * Set the distance threshold within which we've consider centers to have converged.
   * If all centers move less than this Euclidean distance, we stop iterating one run.
   */
  def setEpsilon(epsilon: Double): this.type = {
    this.epsilon = epsilon
    this
  }

  // add ths set partition number
  def setPartitionNumber(p: Double): this.type = {
    this.partitionNumber = p
    this
  }

  // add this log RDD if uncached
  private var warnOnUncachedInput = true
  private def disableUncachedWarning(): this.type = {
    warnOnUncachedInput = false
    this
  }

  /**
   * Train a K-means model on the given set of points; `data` should be cached for high
   * performance, because this is an iterative algorithm.
   */
  def run(data: RDD[Vector]): KmeansModel = {
    // add log uncache input
    if(warnOnUncachedInput && data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its" +
        " parent RDD are also uncached.")
    }

    // Compute squared norms and cache them.
    // this have been modified 
    val norms = data.map {v => 
      val vectorValues: Array[Double] = v.toArray
      val vectorBV: BV[Double] = new BDV[Double](vectorValues) 
      breezeNorm(vectorBV, 2.0)
    }
    norms.persist()

    // this have been modified
    val breezeData = data.map {v =>
      val vectorValues: Array[Double] = v.toArray
      val vectorBV: BV[Double] = new BDV[Double](vectorValues)
      vectorBV
      }.zip(norms).map { case (v, norm) =>
      new BreezeVectorWithNorm(v, norm)
    }

    // Add this to get centers first. 
    var centers = if (initializationMode == Kmeans.RANDOM) {
      initRandom(breezeData)
    } else if (initializationMode == Kmeans.PARALLEL) {
      initKmeansParallel(breezeData)
    } else if (initializationMode == Kmeans.PARTITION) {
      initKmeansParallel(breezeData)
    } else {
      initKmeansDiameter(breezeData)
    }

    

    /* print centers */ 
    // delete output file if exist
    val outfile = "C:/code_temp/spark_temp/centersDistInfo.txt"
    val fileTemp = new File(outfile)
    if (fileTemp.exists()) { fileTemp.delete}
    val writer = new PrintWriter(new File("C:/code_temp/spark_temp/centersInfo.txt"))
    writer.println("The initial possible centers:")
    centers(0).foreach(p => writer.println(p.vector(0) +" "+ p.vector(1)))
    writer.println

    var model = runBreeze(breezeData, centers)
    var lastCenters = toBreezeVector(model.clusterCenters)
    writer.println("The initial centers after Kmeans:")
    lastCenters.foreach(p => writer.println(p.vector(0) +" "+ p.vector(1)))
    writer.println
    // val timestp = (System.nanoTime()/1e9).asInstanceOf[Int]
    while(centers(0).size > k) {
      //print out every step clusters partition
      // val printpath = "C:/code_temp/spark_temp/result" + timestp + "-" + centers(0).size
      // model.predict(data).saveAsTextFile(printpath)
      
      var newCenters = toBreezeVector(model.clusterCenters)
      if(Kmeans.getCentersDistList(lastCenters)(0)._2 < Kmeans.getCentersDistList(newCenters)(0)._2) {
        newCenters = lastCenters
      }
      val tmp = Kmeans.shrinkCenters(newCenters, centers(0).size-1)
      centers = Array.fill(runs)(tmp._1)
      lastCenters = centers(0).clone
      model = runBreeze(breezeData, centers)

      writer.println("The merge centers:")
      tmp._2.foreach(p => writer.println(p.vector(0) +" "+ p.vector(1)))
      writer.println
      writer.println("The centers before Kmeans:")
      lastCenters.foreach(p => writer.println(p.vector(0) +" "+ p.vector(1)))
      writer.println
      writer.println("The centers after Kmeans:")
      model.clusterCenters.foreach(p => writer.println(p(0) +" "+ p(1)))
      writer.println
    }
    writer.close()
    /* end print centers */
    
    norms.unpersist()
    // add log uncache input. warn at the end of run as well.
    if(warnOnUncachedInput && data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its" +
        " parent RDD are also uncached.")
    }
    model
  }

  /**
   * Implementation of K-Means using breeze.
   */
  def runBreeze(data: RDD[BreezeVectorWithNorm], centers: Array[Array[BreezeVectorWithNorm]]): 
      KmeansModel = {

    val sc = data.sparkContext

    val initStartTime = System.nanoTime()

    // add this part to get initial center
    var initCenters = Array.fill(runs, k)(new BreezeVectorWithNorm(centers(0)(0).vector, centers(0)(0).norm))
    for(i <- 0 until runs) {
      for(j <- 0 until k) {
        initCenters(i)(j) = centers(i)(j)
      }
    }

    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(s"Initialization with $initializationMode took " + "%.3f".format(initTimeInSeconds) +
      " seconds.")

    val active = Array.fill(runs)(true)
    val costs = Array.fill(runs)(0.0)

    var activeRuns = new ArrayBuffer[Int] ++ (0 until runs)
    var iteration = 0

    val iterationStartTime = System.nanoTime()

    // Execute iterations of Lloyd's algorithm until all runs have converged
    while (iteration < maxIterations && !activeRuns.isEmpty) {
      type WeightedPoint = (BV[Double], Long)
      def mergeContribs(p1: WeightedPoint, p2: WeightedPoint): WeightedPoint = {
        (p1._1 += p2._1, p1._2 + p2._2)
      }

      val activeCenters = activeRuns.map(r => centers(r)).toArray
      val costAccums = activeRuns.map(_ => sc.accumulator(0.0))

      val bcActiveCenters = sc.broadcast(activeCenters)

      // Find the sum and count of points mapping to each center
      val totalContribs = data.mapPartitions { points =>
        val thisActiveCenters = bcActiveCenters.value
        val runs = thisActiveCenters.length
        val k = thisActiveCenters(0).length
        val dims = thisActiveCenters(0)(0).vector.length

        val sums = Array.fill(runs, k)(BDV.zeros[Double](dims).asInstanceOf[BV[Double]])
        val counts = Array.fill(runs, k)(0L)

        points.foreach { point =>
          (0 until runs).foreach { i =>
            val (bestCenter, cost) = Kmeans.findClosest(thisActiveCenters(i), point)
            costAccums(i) += cost
            sums(i)(bestCenter) += point.vector
            counts(i)(bestCenter) += 1
          }
        }

        val contribs = for (i <- 0 until runs; j <- 0 until k) yield {
          ((i, j), (sums(i)(j), counts(i)(j)))
        }
        contribs.iterator
      }.reduceByKey(mergeContribs).collectAsMap()

      // Update the cluster centers and costs for each active run
      for ((run, i) <- activeRuns.zipWithIndex) {
        var changed = false
        var j = 0
        while (j < k) {
          val (sum, count) = totalContribs((i, j))
          if (count != 0) {
            sum /= count.toDouble
            val newCenter = new BreezeVectorWithNorm(sum)
            if (Kmeans.fastSquaredDistance(newCenter, centers(run)(j)) > epsilon * epsilon) {
              changed = true
            }
            centers(run)(j) = newCenter
          }
          j += 1
        }
        if (!changed) {
          active(run) = false
          logInfo("Run " + run + " finished in " + (iteration + 1) + " iterations")
        }
        costs(run) = costAccums(i).value
      }

      activeRuns = activeRuns.filter(active(_))
      iteration += 1
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(s"Iterations took " + "%.3f".format(iterationTimeInSeconds) + " seconds.")

    if (iteration == maxIterations) {
      logInfo(s"KMeans reached the max number of iterations: $maxIterations.")
    } else {
      logInfo(s"KMeans converged in $iteration iterations.")
    }

    val (minCost, bestRun) = costs.zipWithIndex.min

    logInfo(s"The cost for the best run is $minCost.")

    // this part have been modified
    new KmeansModel(centers(bestRun).map(c => new DenseVector(c.vector.toArray)), initCenters(bestRun).map(c => 
      new DenseVector(c.vector.toArray)))
  }

  /**
   * Initialize `runs` sets of cluster centers at random.
   */
  private def initRandom(data: RDD[BreezeVectorWithNorm])
  : Array[Array[BreezeVectorWithNorm]] = {
    // Sample all the cluster centers in one pass to avoid repeated scans
    val sample = data.takeSample(true, runs * k, new XORShiftRandom().nextInt()).toSeq
    Array.tabulate(runs)(r => sample.slice(r * k, (r + 1) * k).map { v =>
      new BreezeVectorWithNorm(v.vector.toDenseVector, v.norm)
    }.toArray)
  }

  // add this part.
  /** 
   * Initialize `runs` sets of cluster centers.  Account for the enhanced K-means method: (N/k)*f
   */
  private def initKmeansPartition(data: RDD[BreezeVectorWithNorm])
  : Array[Array[BreezeVectorWithNorm]] = {
    val sc = data.sparkContext

    // this part have been modified
    // Sample all the cluster centers in one pass to avoid repeated scans
    val sample = data.takeSample(true, runs * k, seed).toSeq
    var centers = Array.tabulate(runs)(r => sample.slice(r * k, (r + 1) * k).map { v =>
      new BreezeVectorWithNorm(v.vector.toDenseVector, v.norm)
    }.toArray)

    val totalNumPoints = data.count
    val minPoints = (totalNumPoints / k) * partitionNumber   // cluster set points should larger than this value
    val active = Array.fill(runs)(true)
    var activeRuns = new ArrayBuffer[Int] ++ (0 until runs)
    var iteration = 0

    // Execute iterations of Lloyd's algorithm until all runs have converged
    while (iteration < initializationSteps && !activeRuns.isEmpty) {
      def mergeContribs(p1: Int, p2: Int): Int = {
        p1 + p2
      }

      val activeCenters = activeRuns.map(r => centers(r)).toArray
      val bcActiveCenters = sc.broadcast(activeCenters)

      // points mapping to each center
      val totalContribs = data.mapPartitions { points =>
        val thisActiveCenters = bcActiveCenters.value
        val runs = thisActiveCenters.length
        val k = thisActiveCenters(0).length

        val counts = Array.fill(runs, k)(0)

        points.foreach { point => 
          (0 until runs).foreach { i =>
            val (bestCenter, cost) = Kmeans.findClosest(thisActiveCenters(i), point)
            counts(i)(bestCenter) += 1
          }
        }

        val contribs = for(i <- 0 until runs; j <- 0 until k) yield {
          ((i, j), counts(i)(j))
        }
        contribs.iterator
      }.reduceByKey(mergeContribs).collectAsMap()

      // check the number of points in each cluster. If the number of points is less than minPoints,
      // start a new iteration, else break
      for ((run, i) <- activeRuns.zipWithIndex) {
        var outliers = false
        var j = 0
        while (j < k) {
          val count = totalContribs((i, j))
          if (count < minPoints) {
            outliers = true
          }
          j += 1
        }
        if (!outliers) {
          active(run) = false
        } 
        else {
          seed = seed + 1
          val sample = data.takeSample(true, k, seed).toSeq
          for(i <- 0 until k) {
            centers(run)(i) = sample(i)
          }
        }
      }

      activeRuns = activeRuns.filter(active(_))
      iteration += 1
    }
    centers
  }

  /**
   * Convert Array[Vector] to Array[BreezeVectorWithNorm]
   */
  private def toBreezeVector(vectorData: Array[Vector]): Array[BreezeVectorWithNorm] = 
    Array(vectorData.map{v => 
        val vectorValues: Array[Double] = v.toArray
        val vectorBV: BV[Double] = new BDV[Double](vectorValues)
        val norm = breezeNorm(vectorBV, 2.0)
        new BreezeVectorWithNorm(vectorBV, norm)
    }: _*)

  /** 
   * Initialize `runs` sets of cluster centers.  Account for the data set diameter.
   */
  private def initKmeansDiameter(data: RDD[BreezeVectorWithNorm])
  : Array[Array[BreezeVectorWithNorm]] = {
    val sc = data.sparkContext

    val arrPoints = data.collect()
    val points = sc.broadcast(arrPoints)
    val clusters = k * 2 // k is the number of clusters
    val (twoPoints, longestDist) = Kmeans.findLongestDist(points.value)
    val sqrtDist = math.sqrt(longestDist)
    val tolerance = if(sqrtDist*0.003 > 1) sqrtDist*0.003 else 1 // about 5.0

    // find the possible centers
    val center = List((twoPoints(0).vector(0) + twoPoints(1).vector(0))/2,
                      (twoPoints(0).vector(1) + twoPoints(1).vector(1))/2)
    val initSlope = Kmeans.getSlope(twoPoints(0), twoPoints(1))
    var initAngle = math.atan(initSlope)
    val increAngle = math.Pi/clusters
    val slopes = new Array[Double](clusters)
    val intercepts = new Array[Double](clusters)
    slopes(0) = initSlope
    1 to clusters-1 foreach (i => slopes(i) = math.tan(initAngle + increAngle * i))
    0 to clusters-1 foreach (i => intercepts(i) = Kmeans.getIntercept(center, slopes(i)))

    var centerPoints = new ArrayBuffer[ArrayBuffer[BreezeVectorWithNorm]]()
    var flag = false
    var rotCount = 0
    do {
      for(i <- 0 to clusters-1) {
        val tmpAbove = new ArrayBuffer[BreezeVectorWithNorm]()
        val tmpBlow = new ArrayBuffer[BreezeVectorWithNorm]()
        points.value.foreach { p =>
          if(math.abs(p.vector(1) - (slopes(i)*p.vector(0)+intercepts(i))) < tolerance) {
            if(p.vector(1) > center(1)) {
              tmpAbove += p
            }
            else {
              tmpBlow += p
            }
          }
        }
        if(!tmpAbove.isEmpty) {
          centerPoints += tmpAbove.sortWith(_.vector(1) < _.vector(1))
        }
        if(!tmpBlow.isEmpty) {
          centerPoints += tmpBlow.sortWith(_.vector(1) > _.vector(1))
        }
      }

      if(centerPoints.length < clusters){
        initAngle += math.Pi/18
        slopes(0) = math.tan(initAngle)
        1 to clusters-1 foreach(i => slopes(i) = math.tan(initAngle + increAngle*i))
        0 to clusters-1 foreach(i => intercepts(i) = Kmeans.getIntercept(center, slopes(i)))
        flag = true
        rotCount += 1
        if(rotCount < 36) {
          centerPoints.clear()
        }
      }
      else {
        flag = false
        rotCount = 0
      }
    } while(flag && rotCount < 36)

    centerPoints = centerPoints.sortWith(_.size > _.size)
    val pcenters = centerPoints.map(r => r((0.8*r.size).asInstanceOf[Int])).take(clusters)
    Array.fill(runs)(pcenters.toArray)
  }

  /**
   * Initialize `runs` sets of cluster centers using the k-means|| algorithm by Bahmani et al.
   * (Bahmani et al., Scalable K-Means++, VLDB 2012). This is a variant of k-means++ that tries
   * to find with dissimilar cluster centers by starting with a random center and then doing
   * passes where more centers are chosen with probability proportional to their squared distance
   * to the current cluster set. It results in a provable approximation to an optimal clustering.
   *
   * The original paper can be found at http://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf.
   */
  def initKmeansParallel(data: RDD[BreezeVectorWithNorm])
  : Array[Array[BreezeVectorWithNorm]] = {
    // Initialize each run's center to a random point
    val seed = new XORShiftRandom().nextInt()
    val sample = data.takeSample(true, runs, seed).toSeq
    val centers = Array.tabulate(runs)(r => ArrayBuffer(sample(r).toDense))

    // On each step, sample 2 * k points on average for each run with probability proportional
    // to their squared distance from that run's current centers
    var step = 0
    while (step < initializationSteps) {
      val bcCenters = data.context.broadcast(centers)
      val sumCosts = data.flatMap { point =>
        (0 until runs).map { r =>
          (r, Kmeans.pointCost(bcCenters.value(r), point))
        }
      }.reduceByKey(_ + _).collectAsMap()
      val chosen = data.mapPartitionsWithIndex { (index, points) =>
        val rand = new XORShiftRandom(seed ^ (step << 16) ^ index)
        points.flatMap { p =>
          (0 until runs).filter { r =>
            rand.nextDouble() < 2.0 * Kmeans.pointCost(bcCenters.value(r), p) * k / sumCosts(r)
          }.map((_, p))
        }
      }.collect()
      chosen.foreach { case (r, p) =>
        centers(r) += p.toDense
      }
      step += 1
    }

    // Finally, we might have a set of more than k candidate centers for each run; weigh each
    // candidate by the number of points in the dataset mapping to it and run a local k-means++
    // on the weighted centers to pick just k of them
    val bcCenters = data.context.broadcast(centers)
    val weightMap = data.flatMap { p =>
      (0 until runs).map { r =>
        ((r, Kmeans.findClosest(bcCenters.value(r), p)._1), 1.0)
      }
    }.reduceByKey(_ + _).collectAsMap()
    val finalCenters = (0 until runs).map { r =>
      val myCenters = centers(r).toArray
      val myWeights = (0 until myCenters.length).map(i => weightMap.getOrElse((r, i), 0.0)).toArray
      LocalKmeans.kmeansPlusPlus(r, myCenters, myWeights, k, 30)
    }

    finalCenters.toArray
  }
}

// this part has been modified
/**
 * Top-level methods for calling K-means clustering.
 */
object Kmeans {

  lazy val EPSILON = {
    var eps = 1.0
    while ((1.0 + (eps / 2.0)) != 1.0){
      eps /= 2.0
    }
    eps
  }
  // Initialization mode names
  val RANDOM = "random"
  val PARALLEL = "k-means||"
  val PARTITION = "partition"
  val DIAMETER = "diameter"

  /**
   * Returns the index of the closest center to the given point, as well as the squared distance.
   */
  def findClosest(
      centers: TraversableOnce[BreezeVectorWithNorm],
      point: BreezeVectorWithNorm): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance: Double = fastSquaredDistance(center, point)
        if (distance < bestDistance) {
          bestDistance = distance
          bestIndex = i
        }
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  /**
   * Returns the farthest two points, as well as the squared distance.
   */
  // two lines vector product: OA X OB, positive for counter-clockwise turn, vice versa
  def cross(pointO: BreezeVectorWithNorm, pointA: BreezeVectorWithNorm, 
        pointB: BreezeVectorWithNorm): Double = {
    (pointA.vector(0)-pointO.vector(0))*(pointB.vector(1)-pointO.vector(1)) - 
        (pointA.vector(1)-pointO.vector(1))*(pointB.vector(0)-pointO.vector(0))
  }

  // sort the points set based on the x-coordinate by non-descending order
  def cmp(pointA: BreezeVectorWithNorm, pointB: BreezeVectorWithNorm): Boolean = {
    if(pointA.vector(0) == pointB.vector(0)) {
      pointA.vector(1) < pointB.vector(1)
    }
    else {
      pointA.vector(0) < pointB.vector(0)
    }
  }

  //Andrew's monotone chain convex hull algorithm
  def convexHull(data: Array[BreezeVectorWithNorm]): Array[BreezeVectorWithNorm] = {
    if(data.size <= 1) {
        return data
    }
    // sort points by x value
    val points = data.sortWith((pointA,pointB) => cmp(pointA,pointB))

    // build the hull
    val lower = new Stack[BreezeVectorWithNorm]()
    points.foreach { p =>
        while(lower.size >= 2 && cross(lower(1), lower(0), p) <= 0) {
            lower.pop()
        }
        lower.push(p)
    }
    val upper = new Stack[BreezeVectorWithNorm]()
    points.reverse.foreach { p =>
        while(upper.size >= 2 && cross(upper(1), upper(0), p) <= 0) {
            upper.pop()
        }
        upper.push(p)
    }
    val lowArr = lower.take(lower.size-1)
    val upArr = upper.take(upper.size-1)
    (lowArr ++ upArr).toArray.reverse
  }

  // using convex-hull get the diameter of points by rotating caliper 
  def findLongestDist(
      data: TraversableOnce[BreezeVectorWithNorm]): 
    (Array[BreezeVectorWithNorm], Double) = {
    var longestDistance = 0.0
    val maxDistPoints = new Array[BreezeVectorWithNorm](2)
    val convex = convexHull(data.toArray)
    val top = convex.size - 1
    var j = 2
    for(i <- 0 until top) { 
      while(cross(convex(i), convex(i+1), convex(j)) 
            < cross(convex(i), convex(i+1), convex(j+1))) {
        j = (j + 1) % top
      }
      var dist1 = fastSquaredDistance(convex(i), convex(j))
      val dist2 = fastSquaredDistance(convex(i+1), convex(j))
      if(dist1 > dist1) {
        if(longestDistance < dist1) {
          longestDistance = dist1
          maxDistPoints(0) = convex(i)
          maxDistPoints(1) = convex(j)
        }
      }
      else{
        if(longestDistance < dist2) {
          longestDistance = dist2
          maxDistPoints(0) = convex(i+1)
          maxDistPoints(1) = convex(j)
        }
      }
    }
    (maxDistPoints, longestDistance)
  }

  /**
   * Calculate line slope and intercept for two points.
   */
  def getSlope(point1: BreezeVectorWithNorm, point2: BreezeVectorWithNorm): Double = 
    (point2.vector(1)-point1.vector(1)) / (point2.vector(0)-point1.vector(0))

  def getIntercept(point: List[Double], slope: Double): Double = 
    point(1) - slope * point(0)

  /**
  * Get centers pair distance list
  */
  def getCentersDistList(centers: Array[BreezeVectorWithNorm]):
        ArrayBuffer[(List[Int], Double)] = {
    
    val newCenters = ArrayBuffer(centers: _*) 
    val distArray = new ArrayBuffer[(List[Int], Double)]()
    for(i <- 0 until newCenters.size-1) {
      for(j <- i+1 until newCenters.size) {  
        distArray.append((List(i,j), math.sqrt(fastSquaredDistance(newCenters(i),newCenters(j)))))
      }
    }  
    distArray.sortWith(_._2 < _._2)  
  }

  /**
   * Shrink cluster centers with target size.
   */
  def shrinkCenters(centers: Array[BreezeVectorWithNorm], k: Int):
        (Array[BreezeVectorWithNorm] ,Array[BreezeVectorWithNorm]) = {

    val newCenters = ArrayBuffer(centers: _*) 
    val mergePoints = new Array[BreezeVectorWithNorm](2)
    val distArray = getCentersDistList(centers)

    /* print centers distance info */
    val outfile = "C:/code_temp/spark_temp/centersDistInfo.txt"
    val writer = new PrintWriter(new FileOutputStream(new File(outfile), true))
    distArray.foreach(i => writer.println("("+i._1(0)+","+i._1(1)+")"+" : "+i._2))
    writer.println
    writer.close()
    /* end print */

    while(newCenters.size > k) {
      var (p1, p2) = (distArray(0)._1(0), distArray(0)._1(1))
      val newpointV = (newCenters(p1).vector + newCenters(p2).vector)/2.0
      val newpoint = new BreezeVectorWithNorm(newpointV)
      mergePoints(0) = newCenters(p1)
      mergePoints(1) = newCenters(p2)
      newCenters -= (newCenters(p1), newCenters(p2))
      newCenters.append(newpoint)
    }
    (newCenters.toArray, mergePoints)
  }

  /**
   * Returns the K-means cost of a given point against the given cluster centers.
   */
  def pointCost(
      centers: TraversableOnce[BreezeVectorWithNorm],
      point: BreezeVectorWithNorm): Double =
    findClosest(centers, point)._2

  /**
   * Returns the squared Euclidean distance between two vectors computed by
   * [[org.apache.spark.mllib.util.MLUtils#fastSquaredDistance]].
   */
  def fastSquaredDistance(
      v1: BreezeVectorWithNorm,
      v2: BreezeVectorWithNorm): Double = {
    val precision: Double = 1e-6
    val vector1: BV[Double] = v1.vector
    val vector2: BV[Double] = v2.vector
    val norm1 = v1.norm
    val norm2 = v2.norm
    val n = v1.size
    require(v2.size == n)
    require(norm1 >= 0.0 && norm2 >= 0.0)
    val sumSquaredNorm = norm1 * norm1 + norm2 * norm2
    val normDiff = norm1 - norm2
    var sqDist = 0.0
    val precisionBound1 = 2.0 * EPSILON * sumSquaredNorm / (normDiff * normDiff + EPSILON)
    if (precisionBound1 < precision) {
      sqDist = sumSquaredNorm - 2.0 * vector1.dot(vector2)
    } else if (vector1.isInstanceOf[BSV[Double]] || vector2.isInstanceOf[BSV[Double]]) {
      val dot = vector1.dot(vector2)
      sqDist = math.max(sumSquaredNorm - 2.0 * dot, 0.0)
      val precisionBound2 = EPSILON * (sumSquaredNorm + 2.0 * math.abs(dot)) / (sqDist + EPSILON)
      if (precisionBound2 > precision) {
        sqDist = breezeSquaredDistance(vector1, vector2)
      }
    } else {
      sqDist = breezeSquaredDistance(vector1, vector2)
    }
    sqDist
  }
}

/**
 * A breeze vector with its norm for fast distance computation.
 */
class BreezeVectorWithNorm(val vector: BV[Double], val norm: Double) extends Serializable {

  def this(vector: BV[Double]) = this(vector, breezeNorm(vector, 2.0))

  def this(array: Array[Double]) = this(new BDV[Double](array))

  def this(v: Vector) = this{
    val vectorValues: Array[Double] = v.toArray
    val vectorBV: BV[Double] = new BDV[Double](vectorValues)
    vectorBV
  }
  /** Converts the vector to a dense vector. */
  def toDense = new BreezeVectorWithNorm(vector.toDenseVector, norm)
  def size: Int = vector.size
}
