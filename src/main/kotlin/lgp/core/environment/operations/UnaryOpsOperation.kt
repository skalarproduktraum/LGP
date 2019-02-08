package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.ArityException
import lgp.core.program.instructions.ParameterMutateable
import lgp.core.program.instructions.RegisterIndex
import lgp.core.program.instructions.UnaryOperation
import lgp.core.program.registers.Arguments
import net.imagej.DefaultDataset
import net.imagej.ImgPlus
import net.imagej.ops.OpInfo
import net.imglib2.IterableInterval
import net.imglib2.img.Img
import net.imglib2.img.array.ArrayImgFactory
import net.imglib2.type.logic.BitType
import net.imglib2.type.numeric.RealType
import net.imglib2.type.numeric.integer.UnsignedByteType

class UnaryOpsOperation<T: Image>(val opInfo: OpInfo, override val parameters: List<Any> = emptyList(), val requiresInOut: Boolean = false) : UnaryOperation<T>({ args: Arguments<T> ->
    try {
        val start = System.nanoTime()
        ImageJOpsOperationLoader.printlnMaybe("${Thread.currentThread().name}: Running unary op ${opInfo.name} (${opInfo.inputs().joinToString { it.type.simpleName }} -> ${opInfo.outputs().joinToString { it.type.simpleName }}), parameters: ${parameters.joinToString(",")}")
        val arguments = mutableListOf<Any>()

        if(requiresInOut) {
            val ii = args.get(0).image as IterableInterval<*>
            val factory = if(opInfo.name.startsWith("threshold.")) {
                ArrayImgFactory(BitType())
            } else {
                ArrayImgFactory(UnsignedByteType())
            }

            val output = factory.create(ii.dimension(0), ii.dimension(1))

            arguments.add(output)
        }

        arguments.add(args.get(0).image)
        arguments.addAll(parameters)

        val opsOutput = ImageJOpsOperationLoader.ops.run(opInfo.name, *(arguments.toTypedArray()))
        val result = if(opInfo.name.startsWith("threshold.")) {
            ImageJOpsOperationLoader.ops.run("convert.uint8", arguments.get(0))
        } else {
            opsOutput
        }
        val duration = System.nanoTime() - start
        ImageJOpsOperationLoader.printlnMaybe("${Thread.currentThread().name}: ${opInfo.name} took ${duration / 10e5}ms")

        try {
            if (ImageJOpsOperationLoader.debugOps) {
//                    ImageJFunctions.showFloat(result as RandomAccessibleInterval<FloatType>, opInfo.name)
//                    ui.show(opInfo.name, result)
                val filename = "${Thread.currentThread().name}-${System.currentTimeMillis()}-unary-${opInfo.name}.tiff"
                println("Saving result $result to $filename via ${ImageJOpsOperationLoader.io.getSaver(result, filename)}")
                val ds = DefaultDataset(ImageJOpsOperationLoader.context, ImgPlus.wrap(result as Img<RealType<*>>))
                ImageJOpsOperationLoader.io.save(ds, filename)
            }
        } catch (e: Exception) {
            System.err.println("Exception occured while showing debug image: ${e.cause}")
            e.printStackTrace()
        }

        Image.ImgLib2Image(result as IterableInterval<*>) as T
    } catch (e: Exception) {
        ImageJOpsOperationLoader.printlnMaybe("${Thread.currentThread().name}: Execution of unary ${opInfo.name} failed, returning input image.")
        ImageJOpsOperationLoader.printlnMaybe("${Thread.currentThread().name}: Parameters were: ${parameters.joinToString(",")}")
        if(!ImageJOpsOperationLoader.silent) {
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
        return "r[$destination] = ops.run(\"$representation\", r[${ operands[0] }] ${ImageJOpsOperationLoader.parametersToCode(parameters)})"
    }

    /**
     * Provides information about the module.
     */
    override val information = ModuleInformation(
            description = ImageJOpsOperationLoader.ops.help(opInfo.toString())
    )

    override fun execute(arguments: Arguments<T>): T {
        return when {
            arguments.size() != this.arity.number -> throw ArityException("UnaryOperation takes 1 argument but was given ${arguments.size()}.")
            else -> this.func(arguments)
        }
    }

    override fun mutateParameters(): UnaryOpsOperation<T> {
//            println("Old parameters: ${parameters.joinToString(", ")}")
        val newParameters = ImageJOpsOperationLoader.augmentParameters(opInfo)
//            println("New parameters: ${newParameters.joinToString(", ")}")
        return UnaryOpsOperation<T>(opInfo, newParameters, requiresInOut)
    }

    override fun toString(): String {
        return representation
    }
}
