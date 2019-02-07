package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.Operation
import net.imglib2.algorithm.neighborhood.HyperSphereShape
import net.imglib2.algorithm.neighborhood.RectangleShape
import net.imglib2.algorithm.neighborhood.Shape
import org.bytedeco.javacpp.opencv_core
import org.bytedeco.javacpp.opencv_core.*
import org.reflections.Reflections
import java.io.File
import java.lang.reflect.Method
import kotlin.random.Random

class OpenCVOperationsLoader<T: Image>(val typeFilter: Class<*> = Any::class.java, val operationsFilter: List<String> = emptyList()) : OperationLoader<T> {

    data class UnaryCandidate(
            val name: String,
            val factory: Method,
            val operations: List<Method>,
            val parameters: List<Any>
    )

    data class BinaryCandidate(
            val name: String,
            val operation: Method,
            val parameters: List<Any>
    )
    /**
     * Loads a component.
     *
     * @returns A component of type [TComponent].
     */
    override fun load(): List<Operation<T>> {
        // TODO: Type filter does not work


        val opsFilterFile = File("minimalOps.txt")
        val allowedOps = opsFilterFile.readLines().filter { !it.startsWith("#") }
        printlnOnce("Got ${allowedOps.size} allowed ops from ${opsFilterFile.name}")



        val unaryCandidates = algorithms.mapNotNull {
            val name = it.simpleName
            val methods = it.declaredMethods.filter { it.parameters.size == 2 && it.parameters[0].type == opencv_core.Mat::class.java && it.parameters[1].type == opencv_core.Mat::class.java }

            val factories = it.enclosingClass.methods.filter { method -> method.name == "create$name" }.sortedBy { it.parameterCount }

            if(methods.isEmpty() || factories.isEmpty()) {
                null
            } else {
                val parameters = factories.first().parameters.map { p -> p.type }
                UnaryCandidate(name, factories.first(), methods, parameters)
            }
        }

        println("Unary:")
        val unaryOps = unaryCandidates.flatMap {
            it.operations.map { op ->
                println("${it.name}: $op")
                UnaryOpenCVOperation<T>(it.name, it.factory, op, augmentParameters(0, it.factory))
            }
        }

        println("Binary:")
        val binaryCandidates = opencv_core::class.java.declaredMethods.filter {
            it.parameters.size >= 3
                    && it.parameters[0].type == opencv_core.Mat::class.java
                    && it.parameters[1].type == opencv_core.Mat::class.java
                    && it.parameters[2].type == opencv_core.Mat::class.java
                    && !it.parameters.drop(3).any { p -> p.type == opencv_core.Mat::class.java }
                    && !it.parameters.drop(3).any { p -> p.type == opencv_core.GpuMat::class.java }
        }.map {
            println("${it.name}: $it")
            BinaryCandidate(it.name, it, it.parameters.toList())
        }

        val binaryOps = binaryCandidates.map {
            BinaryOpenCVOperation<T>(it.name, it.operation, augmentParameters(3, it.operation))
        }

        return unaryOps + binaryOps
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

        val ref = Reflections("org.bytedeco.javacpp")
        val algorithms = ref.getSubTypesOf(Algorithm::class.java)

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

        fun augmentParameters(skip: Int, factory: Method): List<Any> {
            return factory.parameters.drop(skip).mapIndexedNotNull { i, input ->
                printlnOnce("Requires parameter of ${input.type}")
                when(input.type) {
                    Shape::class.java -> HyperSphereShape(Random.nextLong(0, 10))
                    RectangleShape::class.java -> RectangleShape(Random.nextInt(0, 5), Random.nextBoolean())
                    Boolean::class.java -> Random.nextBoolean()
                    Double::class.java -> Random.nextDouble()
                    java.lang.Double::class.java -> Random.nextDouble()
                    Float::class.java -> Random.nextFloat()
                    java.lang.Float::class.java -> Random.nextFloat()
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
                        System.err.println("Don't know how to construct ${input.type} for parameter $i of ${factory.name}")

                        null
                    }
                }
            }
        }
    }

}
