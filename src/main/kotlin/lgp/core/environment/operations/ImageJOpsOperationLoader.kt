package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.Operation
import lgp.examples.IrisDetectorProblem
import net.imagej.ops.OpInfo
import net.imagej.ops.OpService
import net.imagej.ops.special.BinaryOp
import net.imagej.ops.special.UnaryOp
import net.imglib2.*
import net.imglib2.algorithm.neighborhood.HyperSphereShape
import net.imglib2.algorithm.neighborhood.RectangleShape
import net.imglib2.algorithm.neighborhood.Shape
import net.imglib2.outofbounds.OutOfBoundsFactory
import net.imglib2.outofbounds.OutOfBoundsPeriodicFactory
import net.imglib2.type.numeric.IntegerType
import net.imglib2.type.numeric.NumericType
import net.imglib2.type.numeric.RealType
import net.imglib2.type.numeric.integer.IntType
import net.imglib2.type.numeric.real.FloatType
import org.reflections.Reflections
import org.scijava.io.IOService
import java.io.File
import kotlin.random.Random

class ImageJOpsOperationLoader<T: Image>(val typeFilter: Class<*>, val opsFilter: List<String> = emptyList(), opService: OpService? = null) : OperationLoader<T> {

    enum class OpArity { Unary, Binary, Unknown }
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


        printlnOnce("${unarySubtypes.size} unary classes, ${binarySubtypes.size} binary classes")



        val allOps = ops.infos().filter { it.name in allowedOps }
//        allOps.forEach { op ->
//            println("${op.name}/${op.cInfo().loadDelegateClass()} arity=${unaryOrBinary(op)}" +
//                    "\n\tInputs(${op.inputs().size}): " +
//                    op.inputs().joinToString { "${it.type.simpleName}${if(it.isRequired) { "*" } else {""}}${if(it.isAutoFill) { "(AF)" } else {""}}" } +
//                    "\n\tOutputs(${op.outputs().size}): " +
//                    op.outputs().joinToString { it.type.simpleName })
//        }



        val unaryOps = allOps.filter {
            unaryOrBinary(it) == OpArity.Unary
            && it.inputs().size >= 1
            && it.outputs().size == 1
            && (it.outputs()[0].type.isAssignableFrom(typeFilter) || it.outputs()[0].type.isAssignableFrom(RandomAccessibleInterval::class.java))
            && !it.inputs().any { i -> i.type.simpleName.contains("UnaryComputerOp") || i.type.simpleName.contains("UnaryFunctionOp") || i.type.simpleName.contains("List") }
        }.map {
            val requiresInOut = if(it.name in forcedUnary) {
                it.inputs()[0].name == "out" && it.inputs()[1].name == "in1" && it.inputs()[2].name == "in2"
            } else {
                it.inputs()[0].name == "out" && it.inputs()[1].name == "in"
            }
            printlnOnce("UnaryOp ${it.name} has output type ${it.outputs()[0].type}/${it.outputs()[0].genericType} requiresInOut=$requiresInOut inputs=${it.inputs().joinToString { it.name }} outputs=${it.outputs().joinToString { it.name }}")

            UnaryOpsOperation<T>(it, augmentParameters(it), requiresInOut)
        }

        val binaryOps = allOps.filter {
            unaryOrBinary(it) == OpArity.Binary
            && it.outputs()[0].type.isAssignableFrom(typeFilter)
            && !it.inputs().any { i -> i.type.simpleName.contains("List") }
        }.map {
            val requiresInOut = if(it.inputs().size >= 3) {
                it.inputs()[0].name == "out" && it.inputs()[1].name == "in1" && it.inputs()[2].name == "in2"
            } else {
                false
            }

            printlnOnce("BinaryOp ${it.name} has output type ${it.outputs()[0].type}/${it.outputs()[0].genericType} requiresInOut=$requiresInOut")
            BinaryOpsOperation<T>(it, augmentParameters(it), requiresInOut)
        }

        printlnOnce("Collected ${unaryOps.size} unary ops and ${binaryOps.size} binary ops:")
        printlnOnce("Unary: ${unaryOps.joinToString { "${it.opInfo.name} (${it.opInfo.inputs().size})" }}")
        printlnOnce("Binary: ${binaryOps.joinToString { "${it.opInfo.name} (${it.opInfo.inputs().size})" }}")

        if(unaryOps.isEmpty() || binaryOps.isEmpty()) {
            throw IllegalStateException("Didn't discover either unary or binary ops. Type filtering gone wrong?")
        }

        infosPrinted = true
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
        val context = IrisDetectorProblem.context

        val ops: OpService = context.getService(OpService::class.java) as OpService
        val io = context.getService(IOService::class.java) as IOService

        val ref = Reflections("net.imagej.ops")
        val unarySubtypes = ref.getSubTypesOf(UnaryOp::class.java)
        val binarySubtypes = ref.getSubTypesOf(BinaryOp::class.java)

        val forcedUnary = listOf("morphology.dilate", "morphology.erode", "morphology.topHat")
        val forcedBinary = emptyList<String>()

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

        fun unaryOrBinary(info: OpInfo): OpArity {
            val clazz = info.cInfo().loadDelegateClass()
            return when {
                info.name in forcedUnary || (clazz in unarySubtypes && clazz !in binarySubtypes) -> OpArity.Unary
                info.name in forcedBinary || (clazz in binarySubtypes && clazz !in unarySubtypes) -> OpArity.Binary
                else -> OpArity.Unknown
            }
        }

        fun parametersToCode(parameters: List<Any>): String {
            val parameterString = parameters.joinToString(", ") {
                when (it.javaClass) {
                    HyperSphereShape::class.java -> "HyperSphereShape(${(it as HyperSphereShape).radius})"
                    RectangleShape::class.java -> "RectangleShape(${(it as RectangleShape).span}, ${it.isSkippingCenter})"

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

                    FloatType::class.java -> "${(it as FloatType).get()}"
                    IntegerType::class.java -> "${(it as IntType).get()}"
                    IntType::class.java -> "${(it as IntType).get()}"
                    RealType::class.java -> "${(it as FloatType).get()}"
                    NumericType::class.java -> "${(it as FloatType).get()}"

                    Localizable::class.java -> "Point(${(it as Point).getIntPosition(0)}, ${it.getIntPosition(1)})"
                    OutOfBoundsFactory::class.java -> "OutOfBoundsPeriodicFactory<FloatType, RandomAccessibleInterval<FloatType>>())"
                    OutOfBoundsPeriodicFactory::class.java -> "OutOfBoundsPeriodicFactory<FloatType, RandomAccessibleInterval<FloatType>>())"

                    else -> "(don't know how to code-print: ${it.javaClass.simpleName})"
                }
            }

            return if(parameterString.isEmpty()) {
                ""
            } else {
                ", $parameterString"
            }
        }

        fun augmentParameters(op: OpInfo): List<Any> {
            val cutoff = if(unaryOrBinary(op) == OpArity.Unary) {
                1
            } else {
                2
            }

            return op.inputs().drop(cutoff).mapIndexedNotNull { i, input ->
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
                    NumericType::class.java -> FloatType(Random.nextFloat())
                    RealType::class.java -> FloatType(Random.nextFloat())
                    IntegerType::class.java -> IntType(Random.nextInt(0, 255))
                    Localizable::class.java -> Point(Random.nextInt(-10, 10), Random.nextInt(-10, 10))
                    DoubleArray::class.java -> doubleArrayOf(Random.nextDouble(), Random.nextDouble())
                    OutOfBoundsFactory::class.java -> OutOfBoundsPeriodicFactory<FloatType, RandomAccessibleInterval<FloatType>>()

                    // TODO: see if we can construct these in some cases.
                    IterableInterval::class.java -> null
                    RandomAccessibleInterval::class.java -> null
                    RandomAccessible::class.java -> null

                    // fall-through, show error
                    else -> {
                        System.err.println("Don't know how to construct ${input.type} for parameter $i of ${op.name}")

                        null
                    }
                }
            }
        }

        init {
            unarySubtypes.removeAll(binarySubtypes)
        }
    }

}
