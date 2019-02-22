package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.*
import lgp.core.program.registers.Arguments
import org.bytedeco.javacpp.*
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgproc.*
import org.reflections.Reflections
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.random.Random

fun Mat.get(index: Short): Int {
    return this.ptr(0, index.toInt()).int
}

fun Mat.toList(): List<Int> {
    val elements = ArrayList<Int>(this.rows())
    for(i in 0 until this.rows()) {
        elements.add(this.ptr(0, i).int)
    }

    return elements
}

class OpenCVCUDAOperationsLoader<T: Image> : OperationLoader<T> {

    open class UnaryOpenCVCUDABaseOperation<T: Image>(val name: String, func: (Arguments<T>) -> T, override val parameters: List<Any> = emptyList()) : UnaryOperation<T>(func), ParameterMutateable<T> {
        override fun mutateParameters(): Operation<T> {
            return UnaryOpenCVCUDABaseOperation(name, func, randomParameters(name, parameters))
        }
        /**
         * A way to express an operation in a textual format.
         */
        override val representation: String
            get() = name

        /**
         * Provides a string representation of this operation.
         *
         * @param operands The registers used by the [Instruction] that this [Operation] belongs to.
         * @param destination The destination register of the [Instruction] this [Operation] belongs to.
         */
        override fun toString(operands: List<RegisterIndex>, destination: RegisterIndex): String {
            return "r[$destination] = $representation(r[${ operands[0] }] ${parametersToCode(parameters)})"
        }

        /**
         * Provides information about the module.
         */
        override val information: ModuleInformation
            get() = ModuleInformation(description = representation)

        override fun execute(arguments: Arguments<T>): T {
            printlnMaybe("${Thread.currentThread().name}: Running unary op $representation with ${arguments.get(0).image}")
            return super.execute(arguments)
        }

    }

    open class BinaryOpenCVCUDABaseOperation<T: Image>(val name: String, func: (Arguments<T>) -> T, override val parameters: List<Any> = emptyList()) : BinaryOperation<T>(func), ParameterMutateable<T> {
        override fun mutateParameters(): Operation<T> {
            return BinaryOpenCVCUDABaseOperation(name, func, randomParameters(name, parameters))
        }

        /**
         * A way to express an operation in a textual format.
         */
        override val representation: String
            get() = name

        /**
         * Provides a string representation of this operation.
         *
         * @param operands The registers used by the [Instruction] that this [Operation] belongs to.
         * @param destination The destination register of the [Instruction] this [Operation] belongs to.
         */
        override fun toString(operands: List<RegisterIndex>, destination: RegisterIndex): String {
            return "r[$destination] = $representation(r[${ operands[0] }], r[${ operands[1] }], ${parametersToCode(parameters)})"
        }

        /**
         * Provides information about the module.
         */
        override val information: ModuleInformation
            get() = ModuleInformation(description = representation)

        override fun execute(arguments: Arguments<T>): T {
            printlnMaybe("${Thread.currentThread().name}: Running binary op $representation with ${arguments.get(0).image} and ${arguments.get(1).image}")
            return super.execute(arguments)
        }

    }


    class OpenCVCUDAAdd<T: Image> : BinaryOpenCVCUDABaseOperation<T>("add", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaarithm.add(args.get(0).image as GpuMat, args.get(1).image as GpuMat, result)
        Image.OpenCVGPUImage(result) as T
    })

    class OpenCVCUDASubtract<T: Image> : BinaryOpenCVCUDABaseOperation<T>("subtract", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaarithm.subtract(args.get(0).image as GpuMat, args.get(1).image as GpuMat, result)
        Image.OpenCVGPUImage(result) as T
    })

    class OpenCVCUDAMultiply<T: Image> : BinaryOpenCVCUDABaseOperation<T>("multiply", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaarithm.multiply(args.get(0).image as GpuMat, args.get(1).image as GpuMat, result)
        Image.OpenCVGPUImage(result) as T
    })

    class OpenCVCUDADivide<T: Image> : BinaryOpenCVCUDABaseOperation<T>("divide", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaarithm.divide(args.get(0).image as GpuMat, args.get(1).image as GpuMat, result)
        Image.OpenCVGPUImage(result) as T
    })

    class OpenCVCUDAAbsDiff<T: Image> : BinaryOpenCVCUDABaseOperation<T>("absdiff", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaarithm.absdiff(args.get(0).image as GpuMat, args.get(1).image as GpuMat, result)
        Image.OpenCVGPUImage(result) as T
    })

    class OpenCVCUDAMin<T: Image> : BinaryOpenCVCUDABaseOperation<T>("min", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaarithm.min(args.get(0).image as GpuMat, args.get(1).image as GpuMat, result)
        Image.OpenCVGPUImage(result) as T
    })

    class OpenCVCUDAMax<T: Image> : BinaryOpenCVCUDABaseOperation<T>("max", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaarithm.max(args.get(0).image as GpuMat, args.get(1).image as GpuMat, result)
        Image.OpenCVGPUImage(result) as T
    })

    // Unary Operations
    class OpenCVCUDAAddNumber<T: Image>(override val parameters: List<Any> = listOf(1.0)) : UnaryOpenCVCUDABaseOperation<T>("add", { args: Arguments<T> ->
        val result = GpuMat()
        val add = GpuMat((args.get(0).image as GpuMat).rows(), (args.get(0).image as GpuMat).cols(),
            CV_8U, Scalar(parameters[0] as Double))

        opencv_cudaarithm.add(args.get(0).image as GpuMat, add, result)
        add.release()
        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            return OpenCVCUDAAddNumber(listOf(Random.nextDouble(0.0, 255.0)))
        }
    }

    class OpenCVCUDASubtractNumber<T: Image>(override val parameters: List<Any> = listOf(1.0)) : UnaryOpenCVCUDABaseOperation<T>("subtract", { args: Arguments<T> ->
        val result = GpuMat()
        val subtract = GpuMat((args.get(0).image as GpuMat).rows(), (args.get(0).image as GpuMat).cols(),
            CV_8U, Scalar(parameters[0] as Double))

        opencv_cudaarithm.subtract(args.get(0).image as GpuMat, subtract, result)
        subtract.release()
        Image.OpenCVGPUImage(result) as T
    })  {
        override fun mutateParameters(): Operation<T> {
            return OpenCVCUDASubtractNumber(listOf(Random.nextDouble(0.0, 255.0)))
        }
    }

    class OpenCVCUDAMultiplyNumber<T: Image>(override val parameters: List<Any> = listOf(1.0)) : UnaryOpenCVCUDABaseOperation<T>("multiply", { args: Arguments<T> ->
        val result = GpuMat()
        val multiply = GpuMat((args.get(0).image as GpuMat).rows(), (args.get(0).image as GpuMat).cols(),
            CV_8U, Scalar(parameters[0] as Double))

        opencv_cudaarithm.multiply(args.get(0).image as GpuMat, multiply, result)
        multiply.release()
        Image.OpenCVGPUImage(result) as T
    }, parameters) {
        override fun mutateParameters(): Operation<T> {
            return OpenCVCUDAMultiplyNumber(listOf(Random.nextDouble(1.0, 255.0)))
        }
    }

    class OpenCVCUDADivideNumber<T: Image>(override val parameters: List<Any> = listOf(1.0)) : UnaryOpenCVCUDABaseOperation<T>("divide", { args: Arguments<T> ->
        val result = GpuMat()
        val divide = GpuMat((args.get(0).image as GpuMat).rows(), (args.get(0).image as GpuMat).cols(),
            CV_8U, Scalar(parameters[0] as Double))

        opencv_cudaarithm.divide(args.get(0).image as GpuMat, divide, result)
        divide.release()
        Image.OpenCVGPUImage(result) as T
    })  {
        override fun mutateParameters(): Operation<T> {
            return OpenCVCUDADivideNumber(listOf(Random.nextDouble(0.0, 255.0)))
        }
    }

    class OpenCVCUDAMinNumber<T: Image>(override val parameters: List<Any> = listOf(1.0)) : UnaryOpenCVCUDABaseOperation<T>("maxNumber", { args: Arguments<T> ->
        val result = GpuMat()
        val min = GpuMat((args.get(0).image as GpuMat).rows(), (args.get(0).image as GpuMat).cols(),
            CV_8U, Scalar(parameters[0] as Double))

        opencv_cudaarithm.min(args.get(0).image as GpuMat, min, result)
        min.release()
        Image.OpenCVGPUImage(result) as T
    })  {
        override fun mutateParameters(): Operation<T> {
            return OpenCVCUDAMinNumber(listOf(Random.nextDouble(0.0, 255.0)))
        }
    }

    class OpenCVCUDAMaxNumber<T: Image>(override val parameters: List<Any> = listOf(1.0)) : UnaryOpenCVCUDABaseOperation<T>("maxNumber", { args: Arguments<T> ->
        val result = GpuMat()
        val max = GpuMat((args.get(0).image as GpuMat).rows(), (args.get(0).image as GpuMat).cols(),
            CV_8U, Scalar(parameters[0] as Double))

        opencv_cudaarithm.max(args.get(0).image as GpuMat, max, result)
        max.release()
        Image.OpenCVGPUImage(result) as T
    })  {
        override fun mutateParameters(): Operation<T> {
            return OpenCVCUDAMaxNumber(listOf(Random.nextDouble(1.0, 255.0)))
        }
    }

    class OpenCVCUDALog<T: Image>(override val parameters: List<Any> = emptyList()) : UnaryOpenCVCUDABaseOperation<T>("log", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaarithm.log(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    })  {
        override fun mutateParameters(): Operation<T> {
            return this
        }
    }

    class OpenCVCUDAExp<T: Image>(override val parameters: List<Any> = emptyList()) : UnaryOpenCVCUDABaseOperation<T>("exp", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaarithm.exp(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    })  {
        override fun mutateParameters(): Operation<T> {
            return this
        }
    }

    class OpenCVCUDASqrt<T: Image>(override val parameters: List<Any> = emptyList()) : UnaryOpenCVCUDABaseOperation<T>("sqrt", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaarithm.sqrt(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    })  {
        override fun mutateParameters(): Operation<T> {
            return this
        }
    }

    class OpenCVCUDAInvert<T: Image>(override val parameters: List<Any> = emptyList()) : UnaryOpenCVCUDABaseOperation<T>("invert", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaarithm.bitwise_not(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    })  {
        override fun mutateParameters(): Operation<T> {
            return this
        }
    }

    // Thresholding and filters

    class OpenCVCUDAThreshold<T: Image>(override val parameters: List<Any> = listOf(128.0)) : UnaryOpenCVCUDABaseOperation<T>("threshold", { args: Arguments<T> ->
        val result = GpuMat()

        opencv_cudaarithm.threshold(
            args.get(0).image as GpuMat,
            result,
            parameters[0] as Double,
            255.0,
            THRESH_BINARY)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val threshold = Random.nextDouble(1.0, 255.0)
            return OpenCVCUDAThreshold(listOf(threshold))
        }
    }

    class OpenCVCUDAGauss<T: Image>(
        override val parameters: List<Any> = listOf(3, 3, 0.5),
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createGaussianFilter(
            opencv_core.CV_8U,
            opencv_core.CV_8U,
            Size(parameters[0] as Int, parameters[1] as Int),
            parameters[2] as Double
        )) : UnaryOpenCVCUDABaseOperation<T>("gauss", { args: Arguments<T> ->
        val result = GpuMat()
        filter.apply(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val kernelSizeX = Random.nextInt(1, 5) * 2 + 1
            val kernelSizeY = Random.nextInt(1, 5) * 2 + 1
            val sigma = Random.nextDouble(0.5, 4.0)

            return OpenCVCUDAGauss(listOf(kernelSizeX, kernelSizeY, sigma))
        }
    }

    class OpenCVCUDALaplace<T: Image>(
        override val parameters: List<Any> = listOf(3, 1.0),
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createLaplacianFilter(
            opencv_core.CV_8U,
            opencv_core.CV_8U,
            parameters[0] as Int,
            parameters[1] as Double,
            BORDER_WRAP,
            Scalar(0.0)
        )) : UnaryOpenCVCUDABaseOperation<T>("laplace", { args: Arguments<T> ->
        val result = GpuMat()
        filter.apply(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val kernelSize = listOf(1, 3).random()
            val scale = Random.nextDouble(0.0, 4.0)

            return OpenCVCUDALaplace(listOf(kernelSize, scale))
        }
    }

    class OpenCVCUDABoxFilter<T: Image>(
        override val parameters: List<Any> = listOf(3, 3, 1, 1),
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createBoxFilter(
            opencv_core.CV_8U,
            opencv_core.CV_8U,
            Size(parameters[0] as Int, parameters[1] as Int),
            Point(parameters[2] as Int, parameters[3] as Int),
            BORDER_DEFAULT,
            Scalar(0.0)
        )) : UnaryOpenCVCUDABaseOperation<T>("boxFilter", { args: Arguments<T> ->
        val result = GpuMat()
        filter.apply(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val kernelSizeX = Random.nextInt(1, 9)
            val kernelSizeY = Random.nextInt(1, 9)
            val anchorX = Random.nextInt(0, kernelSizeX)
            val anchorY = Random.nextInt(0, kernelSizeY)

            return OpenCVCUDABoxFilter(listOf(kernelSizeX, kernelSizeY, anchorX, anchorY))
        }
    }

    class OpenCVCUDAErode<T: Image>(
        override val parameters: List<Any> = listOf(Mat.ones(3, 3, CV_8U).asMat()),
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createMorphologyFilter(
            MORPH_ERODE,
            opencv_core.CV_8U,
            parameters[0] as Mat
        )) : UnaryOpenCVCUDABaseOperation<T>("erode", { args: Arguments<T> ->
        val result = GpuMat()
        filter.apply(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            // 0 = rect, 1 = ellipse, 2 = cross
            val shape = Random.nextInt(0, 2)
            val size = Size(Random.nextInt(1, 7), Random.nextInt(1, 7))
            val kernel = getStructuringElement(shape, size)

            return OpenCVCUDAErode(listOf(kernel))
        }
    }

    class OpenCVCUDADilate<T: Image>(
        override val parameters: List<Any> = listOf(Mat.ones(3, 3, CV_8U).asMat()),
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createMorphologyFilter(
            MORPH_DILATE,
            opencv_core.CV_8U,
            parameters[0] as Mat
        )) : UnaryOpenCVCUDABaseOperation<T>("dilate", { args: Arguments<T> ->
        val result = GpuMat()
        filter.apply(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            // 0 = rect, 1 = ellipse, 2 = cross
            val shape = Random.nextInt(0, 2)
            val size = Size(Random.nextInt(1, 11), Random.nextInt(1, 11))
            val kernel = getStructuringElement(shape, size)

            return OpenCVCUDAErode(listOf(kernel))
        }
    }

    class OpenCVCUDAMedian<T: Image>(
        override val parameters: List<Any> = listOf(4)
        ) : UnaryOpenCVCUDABaseOperation<T>("median", { args: Arguments<T> ->
        val result = GpuMat()
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createMedianFilter(
            CV_8U,
            parameters[0] as Int)
        filter.apply(args.get(0).image as GpuMat, result)
        filter.close()

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val windowSize = Random.nextInt(3, 40)

            return OpenCVCUDAMedian(listOf(windowSize))
        }
    }

    class OpenCVCUDABoxMax<T: Image>(
        override val parameters: List<Any> = listOf(4)
    ) : UnaryOpenCVCUDABaseOperation<T>("boxMax", { args: Arguments<T> ->
        val result = GpuMat()
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createBoxMaxFilter(
            CV_8U,
            Size(parameters[0] as Int, parameters[0] as Int))
        filter.apply(args.get(0).image as GpuMat, result)
        filter.close()

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val windowSize = Random.nextInt(3, 16)

            return OpenCVCUDABoxMax(listOf(windowSize))
        }
    }

    class OpenCVCUDABoxMin<T: Image>(
        override val parameters: List<Any> = listOf(4)
    ) : UnaryOpenCVCUDABaseOperation<T>("boxMin", { args: Arguments<T> ->
        val result = GpuMat()
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createBoxMinFilter(
            CV_8U,
            Size(parameters[0] as Int, parameters[0] as Int))
        filter.apply(args.get(0).image as GpuMat, result)
        filter.close()

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val windowSize = Random.nextInt(3, 16)

            return OpenCVCUDABoxMin(listOf(windowSize))
        }
    }

    class OpenCVCUDASobel<T: Image>(
        override val parameters: List<Any> = listOf(2, 2, 3),
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createSobelFilter(
            CV_8U, CV_8U,
            parameters[0] as Int,
            parameters[1] as Int,
            parameters[2] as Int,
            1.0,
            BORDER_DEFAULT,
            BORDER_DEFAULT
        )) : UnaryOpenCVCUDABaseOperation<T>("sobel", { args: Arguments<T> ->
        val result = GpuMat()
        filter.apply(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val dx = Random.nextInt(1, 3)
            val dy = Random.nextInt(1, 3)
            val kernelSize = Random.nextInt(0, 3) * 2 + 1

            return OpenCVCUDASobel(listOf(dx, dy, kernelSize))
        }
    }

    class OpenCVCUDAGradient<T: Image>(
        override val parameters: List<Any> = listOf(Mat.ones(3, 3, CV_8U).asMat()),
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createMorphologyFilter(
            MORPH_GRADIENT,
            opencv_core.CV_8U,
            parameters[0] as Mat
        )) : UnaryOpenCVCUDABaseOperation<T>("gradient", { args: Arguments<T> ->
        val result = GpuMat()
        filter.apply(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            // 0 = rect, 1 = ellipse, 2 = cross
            val shape = Random.nextInt(0, 2)
            val size = Size(Random.nextInt(1, 11), Random.nextInt(1, 11))
            val kernel = getStructuringElement(shape, size)

            return OpenCVCUDAGradient(listOf(kernel))
        }
    }

    class OpenCVCUDAOpening<T: Image>(
        override val parameters: List<Any> = listOf(Mat.ones(3, 3, CV_8U).asMat()),
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createMorphologyFilter(
            MORPH_OPEN,
            opencv_core.CV_8U,
            parameters[0] as Mat
        )) : UnaryOpenCVCUDABaseOperation<T>("opening", { args: Arguments<T> ->
        val result = GpuMat()
        filter.apply(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            // 0 = rect, 1 = ellipse, 2 = cross
            val shape = Random.nextInt(0, 2)
            val size = Size(Random.nextInt(3, 11), Random.nextInt(3, 11))
            val kernel = getStructuringElement(shape, size)

            return OpenCVCUDAOpening(listOf(kernel))
        }
    }

    class OpenCVCUDAClosing<T: Image>(
        override val parameters: List<Any> = listOf(Mat.ones(3, 3, CV_8U).asMat()),
        val filter: opencv_cudafilters.Filter = opencv_cudafilters.createMorphologyFilter(
            MORPH_CLOSE,
            opencv_core.CV_8U,
            parameters[0] as Mat
        )) : UnaryOpenCVCUDABaseOperation<T>("closing", { args: Arguments<T> ->
        val result = GpuMat()
        filter.apply(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            // 0 = rect, 1 = ellipse, 2 = cross
            val shape = Random.nextInt(0, 2)
            val size = Size(Random.nextInt(3, 11), Random.nextInt(3, 11))
            val kernel = getStructuringElement(shape, size)

            return OpenCVCUDAClosing(listOf(kernel))
        }
    }

    class OpenCVCUDACannyEdgeDetector<T: Image>(
        override val parameters: List<Any> = listOf(0.0, 255.0),
        val filter: opencv_cudaimgproc.CannyEdgeDetector = opencv_cudaimgproc.createCannyEdgeDetector(
            parameters[0] as Double,
            parameters[1] as Double
        )) : UnaryOpenCVCUDABaseOperation<T>("cannyEdgeDetect", { args: Arguments<T> ->
        val result = GpuMat()
        filter.detect(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val low = Random.nextDouble(0.0, 254.0)
            val high = Random.nextDouble(low, 255.0)

            return OpenCVCUDACannyEdgeDetector(listOf(low, high))
        }
    }

    class OpenCVCUDABilateral<T: Image>(
        override val parameters: List<Any> = listOf(3, 1.0f, 1.0f)) : UnaryOpenCVCUDABaseOperation<T>("bilateral", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaimgproc.bilateralFilter(args.get(0).image as GpuMat, result,
            parameters[0] as Int,
            parameters[1] as Float,
            parameters[2] as Float)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val kernelSize = Random.nextInt(0, 10) * 2 + 1
            val sigmaColor = Random.nextDouble(0.0, 5.0).toFloat()
            val sigmaSpace = Random.nextDouble(0.0, 5.0).toFloat()

            return OpenCVCUDABilateral(listOf(kernelSize, sigmaColor, sigmaSpace))
        }
    }

    class OpenCVCUDADownsample<T: Image>(
        override val parameters: List<Any> = listOf(2)) : UnaryOpenCVCUDABaseOperation<T>("downsample", { args: Arguments<T> ->
        val result = GpuMat()
        val steps = 2.0f.pow(parameters[0] as Int).roundToInt()
        val src = Mat()
        (args.get(0).image as GpuMat).download(src)
        val srcSize = src.size()
        val tmp = Mat(src.rows() / steps, src.cols() / steps)
        val resultLocal = Mat(src.rows(), src.cols())

//        println("scaling from ${srcSize.width()} x ${srcSize.height()} -> ${src.cols()/steps} x ${src.rows()/steps}")
//        println("${src.size().width()}x${src.size().height()} -> ${tmp.size().width()} x ${tmp.size().height()}")
        resize(src, tmp, Size(srcSize.width() / steps, srcSize.height() / steps))
        resize(tmp, resultLocal, srcSize)
        result.upload(resultLocal)

//        println("Result size: ${result.cols()} x ${result.rows()}")

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val steps = Random.nextInt(1, 5)

            return OpenCVCUDADownsample(listOf(steps))
        }
    }

    class OpenCVCUDAUpsample<T: Image>(
        override val parameters: List<Any> = listOf(2)) : UnaryOpenCVCUDABaseOperation<T>("upsample", { args: Arguments<T> ->
        val result = GpuMat()
        val steps = 2.0f.pow(parameters[0] as Int).roundToInt()
        val src = Mat()
        (args.get(0).image as GpuMat).download(src)
        val srcSize = src.size()
        val tmp = Mat(src.rows() * steps, src.cols() * steps)
        val resultLocal = Mat(src.rows(), src.cols())

//        println("scaling from ${srcSize.width()} x ${srcSize.height()} -> ${src.cols()*steps} x ${src.rows()*steps}")
//        println("${src.size().width()}x${src.size().height()} -> ${tmp.size().width()} x ${tmp.size().height()}")
        resize(src, tmp, Size(srcSize.width() * steps, srcSize.height() * steps))
        resize(tmp, resultLocal, srcSize)
        result.upload(resultLocal)

//        println("Result size: ${result.cols()} x ${result.rows()}")

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val steps = Random.nextInt(1, 5)

            return OpenCVCUDAUpsample(listOf(steps))
        }
    }

    class OpenCVCUDADistanceTransform<T: Image>(
        override val parameters: List<Any> = emptyList()) : UnaryOpenCVCUDABaseOperation<T>("distanceTransform", { args: Arguments<T> ->
        val result = GpuMat()
        val src = Mat()
        (args.get(0).image as GpuMat).download(src)
        val tmp = Mat(src.size())

        // CV_8U result type is only possible for L1 metric
        distanceTransform(src, tmp, DIST_L1, 3, CV_8U)

        result.upload(tmp)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            return this
        }
    }



    class OpenCVCUDADeleteSmallComponents<T: Image>(
        override val parameters: List<Any> = listOf(50)) : UnaryOpenCVCUDABaseOperation<T>("deleteSmallComponents", { args: Arguments<T> ->
        val minSize = parameters[0] as Int
        val result = GpuMat()
        val src = Mat()
        (args.get(0).image as GpuMat).download(src)
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()

        val components = connectedComponentsWithStats(src, labels, stats, centroids, 4, CV_32S)
//        println("Got $components components")
        val sizes = stats.col(CC_STAT_AREA).toList()
        val resultLocal = Mat.zeros(src.size(), CV_8U).asMat()

//        println("sizes are: ${sizes.joinToString(",")}, threshold: $minSize, ${sizes.count { it > minSize }} out of ${sizes.count()} match")
//        val comps = mutableSetOf<Int>()
//        val compsDrawn = mutableSetOf<Int>()

//        val labelsConv = Mat()
//        labels.convertTo(labelsConv, CV_8U, 255.0/components, 0.0)
//        imshow("labels", labelsConv)

        for(y in 0 until labels.rows()) {
            for (x in 0 until labels.cols()) {
                val comp = labels.ptr(y, x).int
//                comps.add(comp)
                if(sizes[comp] >= minSize && comp > 0) {
                    resultLocal.ptr(y, x).put(sizes[comp].toByte())
//                    compsDrawn.add(comp)
                }
            }
        }
//        println("Components were: ${comps.joinToString(",")}")
//        println("Components drawn were: ${compsDrawn.joinToString { "$it (${sizes[it.toInt()]})"}} (${compsDrawn.size} total)")
//
        result.upload(resultLocal)

        Image.OpenCVGPUImage(result) as T
    }) {
        override fun mutateParameters(): Operation<T> {
            val minSize = Random.nextInt(1, 200)

            return OpenCVCUDADeleteSmallComponents(listOf(minSize))
        }
    }


    class OpenCVCUDAHistogramEqualisation<T: Image>(
        override val parameters: List<Any> = emptyList()) : UnaryOpenCVCUDABaseOperation<T>("histogramEqualisation", { args: Arguments<T> ->
        val result = GpuMat()
        opencv_cudaimgproc.equalizeHist(args.get(0).image as GpuMat, result)

        Image.OpenCVGPUImage(result) as T
    })



    /**
     * Loads a component.
     *
     * @returns A component of type [TComponent].
     */
    override fun load(): List<Operation<T>> {

        val operations = algorithms.map {
            val inst = it.getDeclaredConstructor().newInstance()

            inst as Operation<T>
        }

        println("Collected OpenCV ops: ${operations.map { it.javaClass.simpleName }.sorted().joinToString(", ")}")

        return operations
    }

    /**
     * Provides information about the module.
     */
    override val information: ModuleInformation
        get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.


    companion object {
        var silent = System.getProperty("EvolveSilent", "false").toBoolean()
        var debugOps = System.getProperty("DebugOps", "false").toBoolean()
        var infosPrinted = false

        val ref = Reflections("lgp.core.environment.operations")
        val algorithms = ref.getSubTypesOf(UnaryOpenCVCUDABaseOperation::class.java) + ref.getSubTypesOf(BinaryOpenCVCUDABaseOperation::class.java)

        fun printlnOnce(message: Any?) {
            if(!infosPrinted) {
                println(message)
            }
        }

        fun printlnMaybe(message: Any?) {
            if(!silent) {
                println(message)
            }
        }

        fun parametersToCode(parameters: List<Any>): String {
            val parameterString = parameters.joinToString(", ") {
                when (it.javaClass) {
                    java.lang.Boolean::class.java -> it.toString()
                    Boolean::class.java -> it.toString()

                    Double::class.java -> it.toString()
                    java.lang.Double::class.java -> it.toString()
                    DoubleArray::class.java -> "doubleArrayOf(${(it as DoubleArray)[0]}, ${it[1]})"

                    Float::class.java -> "${it}f"
                    java.lang.Float::class.java -> "${it}f"

                    Int::class.java -> it.toString()
                    java.lang.Integer::class.java -> it.toString()
                    Long::class.java -> it.toString()
                    java.lang.Long::class.java -> it.toString()
                    Byte::class.java -> it.toString()
                    java.lang.Byte::class.java -> it.toString()
                    Short::class.java -> it.toString()
                    java.lang.Short::class.java -> it.toString()

                    Mat::class.java -> (it as Mat).toString()

                    else -> "(don't know how to code-print: ${it.javaClass.simpleName})"
                }
            }

            return if(parameterString.isEmpty()) {
                ""
            } else {
                ", $parameterString"
            }
        }

        fun randomParameters(name: String, parameters: List<Any>): List<Any> {
            return parameters.mapIndexedNotNull { i, input ->
                when(input.javaClass) {
                    Boolean::class.java -> Random.nextBoolean()
                    Double::class.java -> Random.nextDouble()
                    java.lang.Double::class.java -> Random.nextDouble()
                    Float::class.java -> Random.nextFloat()
                    java.lang.Float::class.java -> Random.nextFloat()
                    java.lang.Integer::class.java -> Random.nextInt()
                    Int::class.java -> Random.nextInt(0, 5)
                    Long::class.java -> Random.nextLong(0, 5)
                    Byte::class.java -> Random.nextInt(0, 255).toByte()
                    DoubleArray::class.java -> doubleArrayOf(Random.nextDouble(), Random.nextDouble())

                    opencv_core.Mat::class.java -> {
                        val m = opencv_core.Mat(4, 4, CV_8U)
                        randn(m, Mat.zeros(4, 4, CV_8U).asMat(), Mat.ones(4, 4, CV_8U).asMat())
                        m
                    }

                    // fall-through, show error
                    else -> {
                        System.err.println("Don't know how to construct ${input.javaClass} for parameter $i of $name")

                        null
                    }
                }
            }
        }
    }

}
