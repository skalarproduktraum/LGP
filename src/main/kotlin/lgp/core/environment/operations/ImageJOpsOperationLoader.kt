package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.*
import lgp.core.program.registers.Arguments
import net.imagej.ImageJService
import net.imagej.ops.OpInfo
import net.imagej.ops.OpService
import net.imagej.ops.OpUtils
import org.scijava.Context
import org.scijava.module.Module
import org.scijava.service.SciJavaService
import org.scijava.thread.ThreadService

class ImageJOpsOperationLoader<T>(val typeFilter: Class<*>, val opsFilter: List<String> = emptyList()) : OperationLoader<T> {

    class UnaryOpsOperation<T>(val opInfo: OpInfo) : UnaryOperation<T>({ args: Arguments<T> ->
        val output = args.get(0)
        val op = ops.module(opInfo.name, args.get(0))
        op.run()

        output
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

    class BinaryOpsOperation<T>(val opInfo: OpInfo): BinaryOperation<T>({ args: Arguments<T> ->
        val output = args.get(0)
        val op = ops.module(opInfo.name, args.get(0), args.get(1))
        op.run()

        output
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

        val allOps = ops.infos()
        val unaryOps = allOps.filter {
            it.inputs().size == 1
                    && it.outputs().size == 1
                    && it.outputs()[0].type.interfaces.contains(typeFilter)
        }.map {
            println("UnaryOp ${it.name} has output type ${it.outputs()[0].type}/${it.outputs()[0].genericType}")
            UnaryOpsOperation<T>(it)
        }

        val binaryOps = allOps.filter {
            it.inputs().size == 2
                    && it.outputs().size == 1
                    && it.outputs()[0].type.interfaces.contains(typeFilter)
        }.map {
            println("BinaryOp ${it.name} has output type ${it.outputs()[0].type}/${it.outputs()[0].genericType}")
            BinaryOpsOperation<T>(it)
        }

        println("Collected ${unaryOps.size} unary ops and ${binaryOps.size} binary ops")
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