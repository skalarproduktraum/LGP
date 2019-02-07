package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.*
import lgp.core.program.registers.Arguments
import net.imglib2.algorithm.neighborhood.HyperSphereShape
import net.imglib2.algorithm.neighborhood.RectangleShape
import net.imglib2.algorithm.neighborhood.Shape
import org.bytedeco.javacpp.opencv_core
import org.bytedeco.javacpp.opencv_core.Algorithm
import org.reflections.Reflections
import java.io.File
import java.lang.reflect.Method
import kotlin.random.Random

class OpenCVOperationsLoader<T: Image>(val typeFilter: Class<*> = Any::class.java, val operationsFilter: List<String> = emptyList()) : OperationLoader<T> {

    class UnaryOpenCVOperation<T: Image>(val name: String, val factory: Method, val method: Method, override val parameters: List<Any> = emptyList()) : UnaryOperation<T>({ args: Arguments<T> ->
        try {
            val algorithmInstance = factory.invoke(null)
            val result = opencv_core.Mat()
            method.invoke(algorithmInstance, args.get(0).image, result)

            Image.OpenCVImage(result) as T
        } catch (e: Exception) {
            printlnMaybe("${Thread.currentThread().name}: Execution of unary $name failed, returning input image.")
            printlnMaybe("${Thread.currentThread().name}: Parameters were: ${parameters.joinToString(",")}")
            if(!silent) {
                e.printStackTrace()
            }

            args.get(0)
        }
    }), ParameterMutateable<T> {
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
            return "r[$destination] = ops.run(\"$representation\", r[${ operands[0] }] ${parametersToCode(parameters)})"
        }

        /**
         * Provides information about the module.
         */
        override val information = ModuleInformation(
                description = ""
        )

        override fun execute(arguments: Arguments<T>): T {
            return when {
                arguments.size() != this.arity.number -> throw ArityException("UnaryOperation takes 1 argument but was given ${arguments.size()}.")
                else -> this.func(arguments)
            }
        }

        override fun mutateParameters(): UnaryOpenCVOperation<T> {
            return UnaryOpenCVOperation<T>(name, factory, method, augmentParameters(factory))
        }

        override fun toString(): String {
            return representation
        }
    }

    class BinaryOpenCVOperation<T: Image>(val name: String, val method: Method, override val parameters: List<Any> = emptyList()): BinaryOperation<T>({ args: Arguments<T> ->
        val result = opencv_core.Mat()
        val ret = method.invoke(args.get(0), args.get(1), result)

        result as T
    }
    ), ParameterMutateable<T> {
        override val representation: String
            get() = name

        override fun toString(operands: List<RegisterIndex>, destination: RegisterIndex): String {
            return "r[$destination] = ops.run(\"$representation\", r[${operands[0]}], r[${operands[1]}] ${parametersToCode(parameters)})"
        }

        override val information = ModuleInformation(
                description = ""
        )

        override fun mutateParameters(): BinaryOpenCVOperation<T> {
            return BinaryOpenCVOperation<T>(name, method)
        }

        override fun toString(): String {
            return representation
        }
    }

    data class UnaryCandidate(
            val name: String,
            val factory: Method,
            val operations: List<Method>,
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
                UnaryOpenCVOperation<T>(it.name, it.factory, op, augmentParameters(it.factory))
            }
        }

        println("Binary:")
        val binaryCandidates = opencv_core::class.java.declaredMethods.filter {
            it.parameters.size >= 3
                    && it.parameters[0].type == opencv_core.Mat::class.java
                    && it.parameters[1].type == opencv_core.Mat::class.java
                    && it.parameters[2].type == opencv_core.Mat::class.java
        }.map {
            println("${it.name}: $it")
            it.name to it
        }

        val binaryOps = binaryCandidates.map {
            BinaryOpenCVOperation<T>(it.first, it.second)
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

        fun augmentParameters(factory: Method): List<Any> {
            return factory.parameters.mapIndexedNotNull { i, input ->
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
