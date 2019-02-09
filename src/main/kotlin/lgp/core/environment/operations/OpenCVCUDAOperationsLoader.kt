package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.*
import lgp.core.program.registers.Arguments
import org.bytedeco.javacpp.opencv_core
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgproc
import org.bytedeco.javacpp.opencv_ximgproc
import org.reflections.Reflections
import kotlin.random.Random

class OpenCVCUDAOperationsLoader<T: Image>(val typeFilter: Class<*> = Any::class.java, val operationsFilter: List<String> = emptyList()) : OperationLoader<T> {

    open class UnaryOpenCVBaseOperation<T: Image>(val name: String, func: (Arguments<T>) -> T, override val parameters: List<Any> = emptyList()) : UnaryOperation<T>(func), ParameterMutateable<T> {
        override fun mutateParameters(): Operation<T> {
            return UnaryOpenCVBaseOperation(name, func, randomParameters(name, parameters))
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

    open class BinaryOpenCVBaseOperation<T: Image>(val name: String, func: (Arguments<T>) -> T, override val parameters: List<Any> = emptyList()) : BinaryOperation<T>(func), ParameterMutateable<T> {
        override fun mutateParameters(): Operation<T> {
            return BinaryOpenCVBaseOperation(name, func, randomParameters(name, parameters))
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
            return "r[$destination] = $representation(r[${ operands[0] }], r[${ operands[1] }, ${parametersToCode(parameters)})"
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


    class OpenCVAdd<T: Image>() : BinaryOpenCVBaseOperation<T>("add", { args: Arguments<T> ->
        val result = opencv_core.add(args.get(0).image as Mat, args.get(1).image as Mat).asMat()
        Image.OpenCVImage(result) as T
    })

    class OpenCVSubtract<T: Image>() : BinaryOpenCVBaseOperation<T>("subtract", { args: Arguments<T> ->
        val result = opencv_core.subtract(args.get(0).image as Mat, args.get(1).image as Mat).asMat()
        Image.OpenCVImage(result) as T
    })

    class OpenCVMultiply<T: Image>() : BinaryOpenCVBaseOperation<T>("multiply", { args: Arguments<T> ->
        val result = opencv_core.divide(args.get(0).image as Mat, args.get(1).image as Mat).asMat()
        Image.OpenCVImage(result) as T
    })

    class OpenCVDivide<T: Image>() : BinaryOpenCVBaseOperation<T>("divide", { args: Arguments<T> ->
        val result = opencv_core.divide(args.get(0).image as Mat, args.get(1).image as Mat).asMat()
        Image.OpenCVImage(result) as T
    })

    // Unary Operations

    class OpenCVAddNumber<T: Image>(override val parameters: List<Any> = listOf(1.0)) : UnaryOpenCVBaseOperation<T>("add", { args: Arguments<T> ->
        val result = opencv_core.add(args.get(0).image as Mat, Scalar(parameters[0] as Double)).asMat()
        Image.OpenCVImage(result) as T
    })

    class OpenCVSubtractNumber<T: Image>(override val parameters: List<Any> = listOf(1.0)) : UnaryOpenCVBaseOperation<T>("subtract", { args: Arguments<T> ->
        val result = opencv_core.add(args.get(0).image as Mat, Scalar(parameters[0] as Double)).asMat()
        Image.OpenCVImage(result) as T
    })

    class OpenCVMultiplyNumber<T: Image>(override val parameters: List<Any> = listOf(1.0)) : UnaryOpenCVBaseOperation<T>("multiply", { args: Arguments<T> ->
        val result = opencv_core.add(args.get(0).image as Mat, Scalar(parameters[0] as Double)).asMat()
        Image.OpenCVImage(result) as T
    }, parameters)

    class OpenCVDivideNumber<T: Image>(override val parameters: List<Any> = listOf(1.0)) : UnaryOpenCVBaseOperation<T>("divide", { args: Arguments<T> ->
        val result = opencv_core.divide(args.get(0).image as Mat, parameters[0] as Double).asMat()
        Image.OpenCVImage(result) as T
    })

//    class OpenCVCannyEdgeDetector<T: Image>(override val parameters: List<Any> = listOf(0.0, 1.0)) : UnaryOpenCVBaseOperation<T>("cannyEdgeDetector", {args: Arguments<T> ->
//        val edgeDetector = createCannyEdgeDetector(parameters[0] as Double, parameters[1] as Double)
//        val m = Mat()
//        edgeDetector.detect(args.get(0).image as Mat, m)
//
//        Image.OpenCVImage(m) as T
//    })

    class OpenCVL0Smooth<T: Image>(override val parameters: List<Any> = emptyList()) : UnaryOpenCVBaseOperation<T>("l0smooth", {args: Arguments<T> ->
        val m = Mat()
        opencv_ximgproc.l0Smooth(args.get(0).image as Mat, m)

        Image.OpenCVImage(m) as T
    })

//    class OpenCVFastHoughTransform<T: Image>(override val parameters: List<Any> = listOf(1)) : UnaryOpenCVBaseOperation<T>("FastHoughTransform", {args: Arguments<T> ->
//        val m = Mat()
//        opencv_ximgproc.FastHoughTransform(args.get(0).image as Mat, m, parameters[0] as Int)
//
//        Image.OpenCVImage(m) as T
//    })

    class OpenCVThinning<T: Image>(override val parameters: List<Any> = listOf(1)) : UnaryOpenCVBaseOperation<T>("Thinning", {args: Arguments<T> ->
        val m = (args.get(0).image as Mat).clone()
        val converted = Mat()
        (args.get(0).image as Mat).convertTo(converted, CV_8U)

        opencv_ximgproc.thinning(converted, converted)

        val final = Mat()
        converted.convertTo(final, CV_32F)

        Image.OpenCVImage(final) as T
    })

//    class OpenCVAnisotrophicDiffusion<T: Image>(override val parameters: List<Any> = listOf(0.5f, 0.5f, 5)) : UnaryOpenCVBaseOperation<T>("anisotrophicDiffusion", {args: Arguments<T> ->
//        val m = Mat()
//        opencv_ximgproc.anisotropicDiffusion(args.get(0).image as Mat, m, parameters[0] as Float, parameters[1] as Float, parameters[2] as Int)
//
//        Image.OpenCVImage(m) as T
//    })

    class OpenCVBlur<T: Image>(override val parameters: List<Any> = listOf(2, 2)) : UnaryOpenCVBaseOperation<T>("blur", {args: Arguments<T> ->
        val m = Mat()
        opencv_imgproc.blur(args.get(0).image as Mat, m, Size(parameters[0] as Int, parameters[1] as Int))

        Image.OpenCVImage(m) as T
    })

    class OpenCVThreshold<T: Image>(override val parameters: List<Any> = listOf(0.5, 0.5)) : UnaryOpenCVBaseOperation<T>("threshold", {args: Arguments<T> ->
        val m = Mat()
        val converted = Mat()
        (args.get(0).image as Mat).convertTo(converted, CV_8U)
        opencv_imgproc.threshold(converted, m, parameters[0] as Double, parameters[1] as Double, 8)

        val final = Mat()
        m.convertTo(final, CV_32F)

        Image.OpenCVImage(final) as T
    })

//    class OpenCVWatershed<T: Image>(override val parameters: List<Any> = emptyList()) : UnaryOpenCVBaseOperation<T>("watershed", {args: Arguments<T> ->
//        val m = Mat()
//        opencv_imgproc.watershed((args.get(0).image as Mat), m)
//
//        Image.OpenCVImage(m) as T
//    })

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

        println("Collected OpenCV ops: ${operations.joinToString { it.javaClass.simpleName }}")

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
        val algorithms = ref.getSubTypesOf(UnaryOpenCVBaseOperation::class.java) + ref.getSubTypesOf(BinaryOpenCVBaseOperation::class.java)

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
                    Long::class.java -> it.toString()
                    Byte::class.java -> it.toString()

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
