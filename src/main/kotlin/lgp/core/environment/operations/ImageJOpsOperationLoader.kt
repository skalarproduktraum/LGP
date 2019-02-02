package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.*
import lgp.core.program.registers.Arguments
import lgp.examples.IrisDetector
import lgp.examples.IrisDetectorProblem
import lgp.lib.operations.toBoolean
import net.imagej.*
import net.imagej.ops.OpInfo
import net.imagej.ops.OpService
import net.imagej.ops.special.BinaryOp
import net.imagej.ops.special.UnaryOp
import net.imglib2.*
import net.imglib2.algorithm.neighborhood.HyperSphereShape
import net.imglib2.algorithm.neighborhood.RectangleShape
import net.imglib2.algorithm.neighborhood.Shape
import net.imglib2.img.ImagePlusAdapter
import net.imglib2.img.Img
import net.imglib2.img.cell.CellImg
import net.imglib2.img.cell.CellImgFactory
import net.imglib2.img.display.imagej.ImageJFunctions
import net.imglib2.outofbounds.OutOfBoundsFactory
import net.imglib2.outofbounds.OutOfBoundsPeriodicFactory
import net.imglib2.type.logic.BitType
import net.imglib2.type.numeric.IntegerType
import net.imglib2.type.numeric.NumericType
import net.imglib2.type.numeric.RealType
import net.imglib2.type.numeric.integer.IntType
import net.imglib2.type.numeric.real.FloatType
import net.imglib2.view.Views
import org.reflections.Reflections
import org.scijava.Context
import org.scijava.io.IOService
import org.scijava.service.SciJavaService
import org.scijava.thread.ThreadService
import org.scijava.ui.UIService
import java.io.File
import java.util.*
import kotlin.random.Random

class ImageJOpsOperationLoader<T>(val typeFilter: Class<*>, val opsFilter: List<String> = emptyList(), opService: OpService? = null) : OperationLoader<T> {

    class UnaryOpsOperation<T>(val opInfo: OpInfo, val parameters: List<Any> = emptyList(), val requiresInOut: Boolean = false) : UnaryOperation<T>({ args: Arguments<T> ->
        try {
            val start = System.nanoTime()
            printlnMaybe("${Thread.currentThread().name}: Running unary op ${opInfo.name} (${opInfo.inputs().joinToString { it.type.simpleName }} -> ${opInfo.outputs().joinToString { it.type.simpleName }}), parameters: ${parameters.joinToString(",")}")
            val arguments = mutableListOf<Any>()

            if(requiresInOut) {
                val ii = args.get(0) as IterableInterval<*>
                val factory = if(opInfo.name.startsWith("threshold.")) {
                    CellImgFactory(BitType(), 2)
                } else {
                    CellImgFactory(FloatType(), 2)
                }

                val output = factory.create(ii.dimension(0), ii.dimension(1))

                arguments.add(output)
            }

            arguments.add(args.get(0)!!)
            arguments.addAll(parameters)

            val opsOutput = ops.run(opInfo.name, *(arguments.toTypedArray())) as T
            val result = if(opInfo.name.startsWith("threshold.")) {
                ops.run("convert.float32", opsOutput) as T
            } else {
                opsOutput
            }
            val duration = System.nanoTime() - start
            printlnMaybe("${Thread.currentThread().name}: ${opInfo.name} took ${duration/10e5}ms")

            try {
                if (debugOps) {
//                    ImageJFunctions.showFloat(result as RandomAccessibleInterval<FloatType>, opInfo.name)
//                    ui.show(opInfo.name, result)
                    val filename = "${Thread.currentThread().name}-${System.currentTimeMillis()}-unary-${opInfo.name}.tiff"
                    println("Saving result $result to $filename via ${io.getSaver(result, filename)}")
                    val ds = DefaultDataset(context, ImgPlus.wrap(result as Img<RealType<*>>))
                    io.save(ds, filename)
                }
            } catch (e: Exception) {
                System.err.println("Exception occured while showing debug image: ${e.cause}")
                e.printStackTrace()
            }

            result
        } catch (e: Exception) {
            printlnMaybe("${Thread.currentThread().name}: Execution of unary ${opInfo.name} failed, returning input image.")
            printlnMaybe("${Thread.currentThread().name}: Parameters were: ${parameters.joinToString(",")}")
            if(!silent) {
                e.printStackTrace()
            }
            /*val f = if(opInfo.name.startsWith("threshold.")) {
                CellImgFactory(BitType(), 2)
            } else {
                CellImgFactory(FloatType(), 2)
            }

            val input = args.get(0) as IterableInterval<*>
            val img = f.create(input.dimension(0), input.dimension(1))

            img as T
            */
            args.get(0)
        }
    }), ParameterMutateable<T> {
        /**
         * A way to express an operation in a textual format.
         */
        override val representation: String
            get() = opInfo.name

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
            description = ops.help(opInfo.toString())
        )

        override fun execute(arguments: Arguments<T>): T {
            return when {
                arguments.size() != this.arity.number -> throw ArityException("UnaryOperation takes 1 argument but was given ${arguments.size()}.")
                else -> this.func(arguments)
            }
        }

        override fun mutateParameters(): UnaryOpsOperation<T> {
            return UnaryOpsOperation<T>(opInfo, augmentParameters(opInfo), requiresInOut)
        }

        override fun toString(): String {
            return representation
        }
    }

    class BinaryOpsOperation<T>(val opInfo: OpInfo, val parameters: List<Any> = emptyList(), val requiresInOut: Boolean = false): BinaryOperation<T>({ args: Arguments<T> ->
//        val output = args.get(0)
//        val op = ops.module(opInfo.name, args.get(0), args.get(1))
//        op.run()
        try {
            val start = System.nanoTime()
            printlnMaybe("${Thread.currentThread().name}: Running binary op ${opInfo.name} (${opInfo.inputs().joinToString { it.type.simpleName }} -> ${opInfo.outputs().joinToString { it.type.simpleName }}), parameters: ${parameters.joinToString(",")}")
            val arguments = mutableListOf<Any?>()

            if(requiresInOut) {
                val ii = args.get(0) as IterableInterval<*>
                val factory = if(opInfo.name.startsWith("threshold.")) {
                    CellImgFactory(BitType(), 2)
                } else {
                    CellImgFactory(FloatType(), 2)
                }

                val output = if(opInfo.name in nonConformantOps) {
                    null
                } else {
                    factory.create(ii.dimension(0), ii.dimension(1))
                }

                arguments.add(output)
            }

            arguments.add(args.get(0)!!)
            arguments.add(args.get(1)!!)
            arguments.addAll(parameters)

            val result = ops.run(opInfo.name, *(arguments.toTypedArray())) as T

            val duration = System.nanoTime() - start
            printlnMaybe("${Thread.currentThread().name}: ${opInfo.name} took ${duration/10e5}ms")
            try {
                if (debugOps) {
//                    ImageJFunctions.showFloat(result as RandomAccessibleInterval<FloatType>, opInfo.name)
//                    ui.show(opInfo.name, result)
                    val filename = "${Thread.currentThread().name}-${System.currentTimeMillis()}-binary-${opInfo.name}.tiff"
                    println("Saving result $result to $filename via ${io.getSaver(result, filename)}")
                    val ds = DefaultDataset(context, ImgPlus.wrap(result as Img<RealType<*>>))
                    io.save(ds, filename)
                }
            } catch (e: Exception) {
                System.err.println("Exception occured while showing debug image: ${e.cause}")
                e.printStackTrace()
            }

            result
        } catch (e: Exception) {
            printlnMaybe("${Thread.currentThread().name}: Execution of binary ${opInfo.name} failed, returning RHS input image.")
            printlnMaybe("${Thread.currentThread().name}: Parameters were: ${parameters.joinToString(",")}")
            if(!silent) {
                e.printStackTrace()
            }
            /*
            val f = CellImgFactory(FloatType(), 2)
            val input = args.get(0) as IterableInterval<*>
            val img = f.create(input.dimension(0), input.dimension(1))

            img as T
            */
            args.get(1)
        }
    }
    ), ParameterMutateable<T> {
        override val representation: String
        get() = opInfo.name

        override fun toString(operands: List<RegisterIndex>, destination: RegisterIndex): String {
            return "r[$destination] = ops.run(\"$representation\", r[${operands[0]}], r[${operands[1]}] ${parametersToCode(parameters)})"
        }

        override val information = ModuleInformation(
            description = ops.help(opInfo.name)
        )

        override fun mutateParameters(): BinaryOpsOperation<T> {
            return BinaryOpsOperation<T>(opInfo, augmentParameters(opInfo), requiresInOut)
        }

        override fun toString(): String {
            return representation
        }

        companion object {
            val nonConformantOps = listOf("math.divide", "math.subtract", "math.add", "math.multiply")
        }
    }
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
            && it.outputs()[0].type.isAssignableFrom(typeFilter)
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

        val ui: UIService = context.getService(UIService::class.java) as UIService
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
            if (!ui.isVisible) ui.showUI()
        }
    }

}
