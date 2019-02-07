package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.ArityException
import lgp.core.program.instructions.ParameterMutateable
import lgp.core.program.instructions.RegisterIndex
import lgp.core.program.instructions.UnaryOperation
import lgp.core.program.registers.Arguments
import org.bytedeco.javacpp.opencv_core
import java.lang.reflect.Method

class UnaryOpenCVOperation<T: Image>(val name: String, val factory: Method, val method: Method, override val parameters: List<Any> = emptyList()) : UnaryOperation<T>({ args: Arguments<T> ->
    OpenCVOperationsLoader.printlnMaybe("Running Unary OpenCV operation $name with parameters ${parameters.joinToString { it.toString() }}")
    try {
        val algorithmInstance = factory.invoke(null, *parameters.toTypedArray())
        val result = opencv_core.Mat()
        method.invoke(algorithmInstance, args.get(0).image, result)

        Image.OpenCVImage(result) as T
    } catch (e: Exception) {
        OpenCVOperationsLoader.printlnMaybe("${Thread.currentThread().name}: Execution of unary $name failed, returning input image.")
        OpenCVOperationsLoader.printlnMaybe("${Thread.currentThread().name}: Parameters were: ${parameters.joinToString(",")}")
        if(!OpenCVOperationsLoader.silent) {
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
        return "r[$destination] = ops.run(\"$representation\", r[${ operands[0] }] ${OpenCVOperationsLoader.parametersToCode(parameters)})"
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
        return UnaryOpenCVOperation<T>(name, factory, method, OpenCVOperationsLoader.augmentParameters(0, factory))
    }

    override fun toString(): String {
        return representation
    }
}
