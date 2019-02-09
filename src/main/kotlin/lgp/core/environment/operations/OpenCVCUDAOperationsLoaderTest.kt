package lgp.core.environment.operations

import org.bytedeco.javacpp.opencv_core
import org.bytedeco.javacpp.opencv_core.getTickCount
import org.bytedeco.javacpp.opencv_core.getTickFrequency
import org.bytedeco.javacpp.opencv_cudafilters
import org.bytedeco.javacpp.opencv_highgui.*
import org.bytedeco.javacpp.opencv_videoio
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

    @Test
    fun testCUDA() {
        val gauss = opencv_cudafilters.createGaussianFilter(opencv_core.CV_8U, opencv_core.CV_8U, opencv_core.Size(3, 3), 0.5)
        val laplace = opencv_cudafilters.createLaplacianFilter(opencv_core.CV_8U, opencv_core.CV_8U, 4, 2.0, opencv_core.BORDER_DEFAULT, opencv_core.Scalar(0.0))

        val frame = opencv_core.Mat()
        val cap = opencv_videoio.VideoCapture()

        cap.read(frame)
        val gpuImage = opencv_core.GpuMat(frame)

        val result = opencv_core.GpuMat()
        val final = opencv_core.GpuMat()

        var running = true

        while(running) {
            val start = getTickCount()
            cap.read(frame)
            if(frame.empty()) {
                continue
            }

            gpuImage.upload(frame)
            gauss.apply(gpuImage, result)
            laplace.apply(result, final)

            namedWindow("input", WINDOW_NORMAL)
            namedWindow("gauss", WINDOW_NORMAL)
            namedWindow("laplace", WINDOW_NORMAL)

            val finalLocal = opencv_core.Mat()
            val gaussLocal = opencv_core.Mat()

            result.download(gaussLocal)
            final.download(finalLocal)

            val fps = getTickFrequency() / (getTickCount() - start)
            println("fps=$fps")

            imshow("input", frame)
            imshow("gauss", gaussLocal)
            imshow("laplace", finalLocal)

            val key = waitKey(30)
            if(key == 27) {
                running = false
            }
        }

    }
}
