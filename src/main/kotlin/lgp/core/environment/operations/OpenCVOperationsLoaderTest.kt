package lgp.core.environment.operations

import org.bytedeco.javacpp.opencv_core
import org.junit.Test

/**
 * <Description>
 *
 * @author Ulrik Günther <hello></hello>@ulrik.is>
</Description> */
internal class OpenCVOperationsLoaderTest {

    @Test
    fun load() {
        val loader = OpenCVOperationsLoader<Image>(opencv_core.Mat::class.java)
        loader.load()
    }
}
