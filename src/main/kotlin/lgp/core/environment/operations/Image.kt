package lgp.core.environment.operations

import net.imglib2.IterableInterval
import org.bytedeco.javacpp.opencv_core

sealed class Image(open val image: Any) {
    class ImgLib2Image(override val image: IterableInterval<*>): Image(image)
    class OpenCVImage(override val image: opencv_core.Mat): Image(image)
    class OpenCVGPUImage(override val image: opencv_core.GpuMat): Image(image)
}
