package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.*
import lgp.core.program.registers.Arguments
import lgp.lib.operations.toBoolean
import net.imagej.ImageJService
import net.imagej.ops.OpInfo
import net.imagej.ops.OpService
import net.imglib2.Localizable
import net.imglib2.Point
import net.imglib2.algorithm.neighborhood.HyperSphereShape
import net.imglib2.algorithm.neighborhood.Shape
import net.imglib2.type.numeric.NumericType
import net.imglib2.type.numeric.real.FloatType
import org.scijava.Context
import org.scijava.service.SciJavaService
import org.scijava.thread.ThreadService
import java.io.File

class ImageJOpsOperationLoader<T>(val typeFilter: Class<*>, val opsFilter: List<String> = emptyList()) : OperationLoader<T> {

    class UnaryOpsOperation<T>(val opInfo: OpInfo, val parameters: List<Any> = emptyList()) : UnaryOperation<T>({ args: Arguments<T> ->
        println("${Thread.currentThread().id}: Running unary op ${opInfo.name} (${opInfo.inputs().joinToString { it.type.simpleName }} -> ${opInfo.outputs().joinToString { it.type.simpleName }}), parameters: ${parameters.joinToString(",")}")
        val arguments = mutableListOf<Any>()
        arguments.add(args.get(0)!!)
        arguments.addAll(parameters)

        ops.run(opInfo.name, *(arguments.toTypedArray())) as T
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

    class BinaryOpsOperation<T>(val opInfo: OpInfo, val parameters: List<Any> = emptyList()): BinaryOperation<T>({ args: Arguments<T> ->
//        val output = args.get(0)
//        val op = ops.module(opInfo.name, args.get(0), args.get(1))
//        op.run()
        println("${Thread.currentThread().id}: Running binary op ${opInfo.name} (${opInfo.inputs().joinToString { it.type.simpleName }} -> ${opInfo.outputs().joinToString { it.type.simpleName }}), parameters: ${parameters.joinToString(",")}")
        val arguments = mutableListOf<Any>()
        arguments.add(args.get(0)!!)
        arguments.add(args.get(1)!!)
        arguments.addAll(parameters)

        ops.run(opInfo.name, *(arguments.toTypedArray())) as T
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
    /**
     * Loads a component.
     *
     * @returns A component of type [TComponent].
     */
    override fun load(): List<Operation<T>> {
        // TODO: Type filter does not work

        val opsFilterFile = File("minimalOps.txt")
        val allowedOps = opsFilterFile.readLines()
        println("Got ${allowedOps.size} allowed ops from ${opsFilterFile.name}")
        val opsBinaryFilterFile = File("allowedBinaryOps.txt")
        val allowedBinaryOps = opsBinaryFilterFile.readLines()

        val allOps = ops.infos().filter { it.name in allowedOps }
//        allOps.forEach { op ->
//            println("$op.name" +
//                    "\n\tInputs(${op.inputs().size}): " +
//                    op.inputs().joinToString { "${it.type.simpleName}${if(it.isRequired) { "*" } else {""}}${if(it.isAutoFill) { "(AF)" } else {""}}" } +
//                    "\n\tOutputs(${op.outputs().size}): " +
//                    op.outputs().joinToString { it.type.simpleName })
//        }

        val unaryOps = allOps.filter {
            !allowedBinaryOps.contains(it.name) &&
            it.inputs().size >= 1
//                    && it.inputs()[0].type.isAssignableFrom(typeFilter)
                    && it.outputs().size == 1
                    && it.outputs()[0].type.isAssignableFrom(typeFilter)
//                    && !it.inputs().drop(1).any { op -> op.type.isAssignableFrom(typeFilter) }
                    && !it.inputs().any { i -> i.type.simpleName.contains("UnaryComputerOp") || i.type.simpleName.contains("UnaryFunctionOp") }
        }.map {
            println("UnaryOp ${it.name} has output type ${it.outputs()[0].type}/${it.outputs()[0].genericType}")
            println("i0: ${it.inputs()[0]?.ioType} i1: ${it.inputs().getOrNull(1)?.ioType}")
            val parameters = it.inputs().drop(1).mapNotNull { o ->
                println("Requires parameter of ${o.type}")
                when(o.type) {
                    Shape::class.java -> HyperSphereShape((Math.random()*10.0f).toLong())
                    Boolean::class.java -> Math.random().toBoolean()
                    Double::class.java -> Math.random()
                    Float::class.java -> Math.random().toFloat()
                    Int::class.java -> (Math.random() * 10.0f).toInt()
                    Long::class.java -> (Math.random() * 10.0f).toInt()
                    Byte::class.java -> (Math.random() * 255).toByte()
                    NumericType::class.java -> FloatType(Math.random().toFloat())
                    Localizable::class.java -> Point((Math.random() * 2048).toInt(), (Math.random() * 2048).toInt())
                    DoubleArray::class.java -> doubleArrayOf(Math.random(), Math.random())
                    else -> null
                }
            }

            UnaryOpsOperation<T>(it, parameters)
        }

        val binaryOps = allOps.filter {
            allowedBinaryOps.contains(it.name) &&
            it.inputs().size >= 2
//                    && it.inputs()[0].type.isAssignableFrom(typeFilter)
//                    && it.inputs()[1].type.isAssignableFrom(typeFilter)
                    && it.outputs().size == 1
                    && it.outputs()[0].type.isAssignableFrom(typeFilter)
                    && !it.inputs().drop(2).any { op -> op.type.isAssignableFrom(typeFilter) }
        }.map {
            println("BinaryOp ${it.name} has output type ${it.outputs()[0].type}/${it.outputs()[0].genericType}")
            val parameters = it.inputs().drop(2).mapNotNull { o ->
                val t = when(o.type) {
                    Shape::class.java -> HyperSphereShape((Math.random()*10.0f).toLong())
                    Boolean::class.java -> Math.random().toBoolean()
                    Double::class.java -> Math.random()
                    Float::class.java -> Math.random().toFloat()
                    Int::class.java -> (Math.random() * 10.0f).toInt()
                    Long::class.java -> (Math.random() * 10.0f).toInt()
                    Byte::class.java -> (Math.random() * 255).toByte()
                    NumericType::class.java -> FloatType(Math.random().toFloat())
                    Localizable::class.java -> Point((Math.random() * 2048).toInt(), (Math.random() * 2048).toInt())
                    DoubleArray::class.java -> doubleArrayOf(Math.random(), Math.random())
                    else -> null
                }
                println("Requires parameter of ${o.type} -> $t")
                t
            }

            BinaryOpsOperation<T>(it, parameters)
        }

        println("Collected ${unaryOps.size} unary ops and ${binaryOps.size} binary ops")

        if(unaryOps.isEmpty() || binaryOps.isEmpty()) {
            throw IllegalStateException("Didn't discover either unary or binary ops. Type filtering gone wrong?")
        }

        return unaryOps + binaryOps
    }

    /**
     * Provides information about the module.
     */
    override val information: ModuleInformation
        get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.


    companion object {
        val context = Context(
            ImageJService::class.java,
            SciJavaService::class.java,
            ThreadService::class.java,
            OpService::class.java
        )

        val ops: OpService = context.getService(OpService::class.java) as OpService
    }

}
