package lgp.core.environment.operations

import org.bytedeco.javacpp.*
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_highgui.*
import org.bytedeco.javacpp.opencv_imgproc.*
import org.junit.Test

/**
 * <Description>
 *
 * @author Ulrik Günther <hello></hello>@ulrik.is>
</Description> */
class OpenCVCUDAOperationsLoaderTest {

    @Test
    fun load() {
    }

    fun randomCircles(mat: Mat, rng: RNG, count: Int): Mat {
        for(i in 0 until count) {
            circle(
                mat, Point(rng.uniform(0, mat.rows()), rng.uniform(0, mat.cols())),
                rng.uniform(20, 50),
                Scalar(rng.uniform(0.0, 255.0))
            )
        }

        return mat
    }

    @Test
    fun testCUDA() {
        val showWindows = System.getProperty("ShowCUDAWindows", "true").toBoolean()
        val rng = opencv_core.RNG()
        var frame = opencv_core.Mat(480, 640, CV_8U)
        randomCircles(frame, rng, 10)

        val gpuImage = opencv_core.GpuMat(frame)

        println("Creating filters")
        val gauss = opencv_cudafilters.createGaussianFilter(opencv_core.CV_8U, opencv_core.CV_8U, opencv_core.Size(7, 7), 0.5)
        val laplace = opencv_cudafilters.createLaplacianFilter(opencv_core.CV_8U, opencv_core.CV_8U, 3, 2.0, opencv_core.BORDER_DEFAULT, opencv_core.Scalar(0.0))

        val result = opencv_core.GpuMat()
        val final = opencv_core.GpuMat()

        var running = true

        if(showWindows) {
            namedWindow("input", WINDOW_NORMAL)
            namedWindow("gauss", WINDOW_NORMAL)
            namedWindow("laplace", WINDOW_NORMAL)
        }

        println("Running")
        while(running) {
            val start = getTickCount()
            randomCircles(frame, rng, rng.uniform(2, 16))

            if(frame.empty()) {
                println("Frame is empty")
                continue
            }

            gpuImage.upload(frame)
            gauss.apply(gpuImage, result)
            laplace.apply(result, final)
            gauss.apply(final, result)
            gauss.apply(result, final)
            laplace.apply(final, result)
            gauss.apply(result, final)
            opencv_cudaarithm.subtract(result, final, gpuImage)
            opencv_cudaarithm.threshold(gpuImage, result, rng.uniform(0.0, 255.0), 255.0, THRESH_BINARY)
            opencv_cudaarithm.absdiff(gpuImage, result, final)

            val finalLocal = opencv_core.Mat()
            val gaussLocal = opencv_core.Mat()

            result.download(gaussLocal)
            final.download(finalLocal)

            val fps = getTickFrequency() / (getTickCount() - start)
            println("fps=$fps")

            if(showWindows) {
                imshow("input", frame)
                imshow("gauss", gaussLocal)
                imshow("laplace", finalLocal)

                val key = waitKey(30)
                if (key == 27) {
                    running = false
                }
            }
        }

    }
}
