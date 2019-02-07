package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.BinaryOperation
import lgp.core.program.instructions.ParameterMutateable
import lgp.core.program.instructions.RegisterIndex
import lgp.core.program.registers.Arguments
import org.bytedeco.javacpp.opencv_core
import java.lang.reflect.Method

class BinaryOpenCVOperation<T: Image>(val name: String, val method: Method, override val parameters: List<Any> = emptyList()): BinaryOperation<T>({ args: Arguments<T> ->
    OpenCVOperationsLoader.printlnMaybe("Running Binary OpenCV operation $name with parameters ${parameters.joinToString { it.toString() }}")
    val result = opencv_core.Mat()
    val ret = method.invoke(null, args.get(0).image, args.get(1).image, result, *parameters.toTypedArray())

    Image.OpenCVImage(result) as T
}
), ParameterMutateable<T> {
    override val representation: String
        get() = name

    override fun toString(operands: List<RegisterIndex>, destination: RegisterIndex): String {
        return "r[$destination] = ops.run(\"$representation\", r[${operands[0]}], r[${operands[1]}] ${OpenCVOperationsLoader.parametersToCode(parameters)})"
    }

    override val information = ModuleInformation(
            description = ""
    )

    override fun mutateParameters(): BinaryOpenCVOperation<T> {
        return BinaryOpenCVOperation<T>(name, method, OpenCVOperationsLoader.augmentParameters(3, method))
    }

    override fun toString(): String {
        return representation
    }
}
