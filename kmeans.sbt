name := "Kmeans"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies ++= Seq(
	"org.apache.spark" %% "spark-core" % "1.1.1",
	"org.apache.spark" %% "spark-mllib" % "1.1.1"
)

libraryDependencies ++= Seq(
    // other dependencies here 
	"org.scalanlp" %% "breeze" % "0.9",
	// native libraries are not included by default. add this if you want them(as of 0.7)
	// native libraries greatly improve performance, but increase jar sizes.
	"org.scalanlp" %% "breeze-natives" % "0.9"
)

resolvers ++= Seq(
	// other resolvers here
	// if you want to use snapshot builds (currently 0.8-SNAPSHOT), use this.
	"Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
	"Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
