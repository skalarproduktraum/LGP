package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.BinaryOperation
import lgp.core.program.instructions.Operation
import lgp.core.program.instructions.RegisterIndex
import lgp.core.program.instructions.UnaryOperation
import lgp.core.program.registers.Arguments
import net.imagej.ImageJService
import net.imagej.ops.OpService
import net.imagej.ops.OpUtils
import org.scijava.Context
import org.scijava.service.SciJavaService
import org.scijava.thread.ThreadService

class ImageJOpsOperationLoader<T>(val typeFilter: Class<*>, val opsFilter: List<String> = emptyList()) : OperationLoader<T> {
    /**
     * Loads a component.
     *
     * @returns A component of type [TComponent].
     */
    override fun load(): List<Operation<T>> {
        val context = Context(
            ImageJService::class.java,
            SciJavaService::class.java,
            ThreadService::class.java
        )

        val ops: OpService = context.getService(OpService::class.java) as OpService

        val unaryOps = ops.ops().mapNotNull {
            val op = ops.module(it)

            if(OpUtils.inputs(op.info).size == 1
                && OpUtils.outputs(op.info).size == 1
                && OpUtils.inputs(op.info)[0].genericType == typeFilter) {
                val unaryOp = object: UnaryOperation<T>({
                        args: Arguments<T> ->
                        val output = args.get(0)
                        op.initialize()
                        op.setInput(OpUtils.inputs(op.info)[0].name, args.get(0))
                        op.setOutput(OpUtils.outputs(op.info)[0].name, output)
                        op.run()

                        output
                    }
                ) {
                    override val representation: String
                        get() = "[ops:$it]"

                    override fun toString(operands: List<RegisterIndex>, destination: RegisterIndex): String {
                        return "r[$destination] = r[${ operands[0] }]"
                    }

                    override val information = ModuleInformation(
                        description = ops.help(it)
                    )

                }

                unaryOp
            } else {
                null
            }
        }

        val binaryOps = ops.ops().mapNotNull {
            val op = ops.module(it)

            if (OpUtils.inputs(op.info).size == 2
                && OpUtils.outputs(op.info).size == 1
                && OpUtils.inputs(op.info)[0].genericType == typeFilter
            ) {
                val binaryOp = object : BinaryOperation<T>({ args: Arguments<T> ->
                    val output = args.get(0)
                    op.initialize()
                    op.setInput(OpUtils.inputs(op.info)[0].name, args.get(0))
                    op.setInput(OpUtils.inputs(op.info)[1].name, args.get(1))
                    op.setOutput(OpUtils.outputs(op.info)[0].name, output)
                    op.run()

                    output
                }
                ) {
                    override val representation: String
                        get() = "[ops:$it]"

                    override fun toString(operands: List<RegisterIndex>, destination: RegisterIndex): String {
                        return "r[$destination] = r[${operands[0]} $representation ${operands[1]}]"
                    }

                    override val information = ModuleInformation(
                        description = ops.help(it)
                    )

                }

                binaryOp
            } else {
                null
            }
        }

        return unaryOps + binaryOps
    }

    /**
     * Provides information about the module.
     */
    override val information: ModuleInformation
        get() = TODO("not implemented") //To change initializer of created properties use File | Settings | File Templates.

}