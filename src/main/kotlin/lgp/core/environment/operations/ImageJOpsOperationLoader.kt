package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.*
import lgp.core.program.registers.Arguments
import lgp.lib.operations.toBoolean
import net.imagej.ImageJService
import net.imagej.ops.OpInfo
import net.imagej.ops.OpService
import net.imagej.ops.special.BinaryOp
import net.imagej.ops.special.UnaryOp
import net.imglib2.IterableInterval
import net.imglib2.Localizable
import net.imglib2.Point
import net.imglib2.RandomAccessibleInterval
import net.imglib2.algorithm.neighborhood.HyperSphereShape
import net.imglib2.algorithm.neighborhood.Shape
import net.imglib2.img.cell.CellImgFactory
import net.imglib2.outofbounds.OutOfBoundsFactory
import net.imglib2.outofbounds.OutOfBoundsPeriodicFactory
import net.imglib2.type.logic.BitType
import net.imglib2.type.numeric.NumericType
import net.imglib2.type.numeric.RealType
import net.imglib2.type.numeric.real.FloatType
import org.reflections.Reflections
import org.scijava.Context
import org.scijava.service.SciJavaService
import org.scijava.thread.ThreadService
import java.io.File

class ImageJOpsOperationLoader<T>(val typeFilter: Class<*>, val opsFilter: List<String> = emptyList()) : OperationLoader<T> {

    class UnaryOpsOperation<T>(val opInfo: OpInfo, val parameters: List<Any> = emptyList(), val requiresInOut: Boolean = false) : UnaryOperation<T>({ args: Arguments<T> ->
        try {
            println("${Thread.currentThread().id}: Running unary op ${opInfo.name} (${opInfo.inputs().joinToString { it.type.simpleName }} -> ${opInfo.outputs().joinToString { it.type.simpleName }}), parameters: ${parameters.joinToString(",")}")
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
            if(opInfo.name.startsWith("threshold.")) {
                ops.run("convert.float32", opsOutput) as T
            } else {
                opsOutput
            }
        } catch (e: Exception) {
            println("${Thread.currentThread().id}: Execution of unary ${opInfo.name} failed, returning empty image.")
            println("${Thread.currentThread().id}: Parameters were: ${parameters.joinToString(",")}")
            e.printStackTrace()
            val f = if(opInfo.name.startsWith("threshold.")) {
                CellImgFactory(BitType(), 2)
            } else {
                CellImgFactory(FloatType(), 2)
            }

            val input = args.get(0) as IterableInterval<*>
            val img = f.create(input.dimension(0), input.dimension(1))

            img as T
        }
    }) {
        /**
         * A way to express an operation in a textual format.
         */
        override val representation: String
            get() = "ops:${opInfo.name}"

        /**
         * Provides a string representation of this operation.
         *
         * @param operands The registers used by the [Instruction] that this [Operation] belongs to.
         * @param destination The destination register of the [Instruction] this [Operation] belongs to.
         */
        override fun toString(operands: List<RegisterIndex>, destination: RegisterIndex): String {
            return "r[$destination] = r[${ operands[0] }]"
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
    }

    class BinaryOpsOperation<T>(val opInfo: OpInfo, val parameters: List<Any> = emptyList(), requiresInOut: Boolean = false): BinaryOperation<T>({ args: Arguments<T> ->
//        val output = args.get(0)
//        val op = ops.module(opInfo.name, args.get(0), args.get(1))
//        op.run()
        try {
            println("${Thread.currentThread().id}: Running binary op ${opInfo.name} (${opInfo.inputs().joinToString { it.type.simpleName }} -> ${opInfo.outputs().joinToString { it.type.simpleName }}), parameters: ${parameters.joinToString(",")}")
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
            arguments.add(args.get(1)!!)
            arguments.addAll(parameters)

            ops.run(opInfo.name, *(arguments.toTypedArray())) as T
        } catch (e: Exception) {
            println("${Thread.currentThread().id}: Execution of binary ${opInfo.name} failed, returning empty image.")
            println("${Thread.currentThread().id}: Parameters were: ${parameters.joinToString(",")}")
            e.printStackTrace()
            val f = CellImgFactory(FloatType(), 2)
            val input = args.get(0) as IterableInterval<*>
            val img = f.create(input.dimension(0), input.dimension(1))

            img as T
        }
    }
    ) {
        override val representation: String
        get() = "[ops:${opInfo.name}]"

        override fun toString(operands: List<RegisterIndex>, destination: RegisterIndex): String {
            return "r[$destination] = r[${operands[0]} $representation ${operands[1]}]"
        }

        override val information = ModuleInformation(
            description = ops.help(opInfo.name)
        )

    }
    enum class OpArity { Unary, Binary, Unknown }
    /**
     * Loads a component.
     *
     * @returns A component of type [TComponent].
     */
    override fun load(): List<Operation<T>> {
        // TODO: Type filter does not work

        val ref = Reflections("net.imagej.ops")

        val opsFilterFile = File("minimalOps.txt")
        val allowedOps = opsFilterFile.readLines()
        printlnOnce("Got ${allowedOps.size} allowed ops from ${opsFilterFile.name}")

        val unarySubtypes =  ref.getSubTypesOf(UnaryOp::class.java)
        val binarySubtypes =  ref.getSubTypesOf(BinaryOp::class.java)

        unarySubtypes.removeAll(binarySubtypes)

        printlnOnce("${unarySubtypes.size} unary classes, ${binarySubtypes.size} binary classes")


        fun unaryOrBinary(info: OpInfo): OpArity {
            val clazz = info.cInfo().loadDelegateClass()
            return when {
                clazz in unarySubtypes && clazz !in binarySubtypes -> OpArity.Unary
                clazz in binarySubtypes && clazz !in unarySubtypes -> OpArity.Binary
                else -> OpArity.Unknown
            }
        }

        val allOps = ops.infos().filter { it.name in allowedOps }
//        allOps.forEach { op ->
//            println("${op.name}/${op.cInfo().loadDelegateClass()} arity=${unaryOrBinary(op)}" +
//                    "\n\tInputs(${op.inputs().size}): " +
//                    op.inputs().joinToString { "${it.type.simpleName}${if(it.isRequired) { "*" } else {""}}${if(it.isAutoFill) { "(AF)" } else {""}}" } +
//                    "\n\tOutputs(${op.outputs().size}): " +
//                    op.outputs().joinToString { it.type.simpleName })
//        }

        fun augmentParameters(op: OpInfo): List<Any> {
            val cutoff = if(unaryOrBinary(op) == OpArity.Unary) {
                1
            } else {
                2
            }

            return op.inputs().drop(cutoff).mapNotNull { input ->
                printlnOnce("Requires parameter of ${input.type}")
                when(input.type) {
                    Shape::class.java -> HyperSphereShape((Math.random()*10.0f).toLong())
                    Boolean::class.java -> Math.random().toBoolean()
                    Double::class.java -> Math.random()
                    Float::class.java -> Math.random().toFloat()
                    Int::class.java -> (Math.random() * 10.0f).toInt()
                    Long::class.java -> (Math.random() * 10.0f).toInt()
                    Byte::class.java -> (Math.random() * 255).toByte()
                    NumericType::class.java -> FloatType(Math.random().toFloat())
                    RealType::class.java -> FloatType(Math.random().toFloat())
                    Localizable::class.java -> Point((Math.random() * 2048).toInt(), (Math.random() * 2048).toInt())
                    DoubleArray::class.java -> doubleArrayOf(Math.random(), Math.random())
                    OutOfBoundsFactory::class.java -> OutOfBoundsPeriodicFactory<FloatType, RandomAccessibleInterval<FloatType>>()
                    else -> null
                }
            }
        }

        val unaryOps = allOps.filter {
            unaryOrBinary(it) == OpArity.Unary
            && it.inputs().size >= 1
            && it.outputs().size == 1
            && it.outputs()[0].type.isAssignableFrom(typeFilter)
            && !it.inputs().any { i -> i.type.simpleName.contains("UnaryComputerOp") || i.type.simpleName.contains("UnaryFunctionOp") }
        }.map {
            val requiresInOut = it.inputs()[0].name == "out" && it.inputs()[1].name == "in"
            printlnOnce("UnaryOp ${it.name} has output type ${it.outputs()[0].type}/${it.outputs()[0].genericType} requiresInOut=$requiresInOut")

            UnaryOpsOperation<T>(it, augmentParameters(it), requiresInOut)
        }

        val binaryOps = allOps.filter {
            unaryOrBinary(it) == OpArity.Binary
            && it.outputs()[0].type.isAssignableFrom(typeFilter)
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
        var infosPrinted = false
        val context = Context(
            ImageJService::class.java,
            SciJavaService::class.java,
            ThreadService::class.java,
            OpService::class.java
        )

        val ops: OpService = context.getService(OpService::class.java) as OpService

        fun printlnOnce(message: Any?) {
            if(!infosPrinted) {
                println(message)
            }
        }
    }

}
