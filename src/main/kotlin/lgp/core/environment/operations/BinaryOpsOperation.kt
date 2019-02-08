package lgp.core.environment.operations

import lgp.core.modules.ModuleInformation
import lgp.core.program.instructions.BinaryOperation
import lgp.core.program.instructions.ParameterMutateable
import lgp.core.program.instructions.RegisterIndex
import lgp.core.program.registers.Arguments
import net.imagej.DefaultDataset
import net.imagej.ImgPlus
import net.imagej.ops.OpInfo
import net.imglib2.IterableInterval
import net.imglib2.img.Img
import net.imglib2.img.array.ArrayImgFactory
import net.imglib2.type.logic.BitType
import net.imglib2.type.numeric.integer.UnsignedByteType

class BinaryOpsOperation<T: Image>(val opInfo: OpInfo, override val parameters: List<Any> = emptyList(), val requiresInOut: Boolean = false): BinaryOperation<T>({ args: Arguments<T> ->
//        val output = args.get(0)
//        val op = ops.module(opInfo.name, args.get(0), args.get(1))
//        op.run()
    try {
        val start = System.nanoTime()
        ImageJOpsOperationLoader.printlnMaybe("${Thread.currentThread().name}: Running binary op ${opInfo.name} (${opInfo.inputs().joinToString { it.type.simpleName }} -> ${opInfo.outputs().joinToString { it.type.simpleName }}), parameters: ${parameters.joinToString(",")}")
        val arguments = mutableListOf<Any?>()

        if(requiresInOut) {
            val ii = args.get(0).image as IterableInterval<*>
            val factory = if(opInfo.name.startsWith("threshold.")) {
                ArrayImgFactory(BitType())
            } else {
                ArrayImgFactory(UnsignedByteType())
            }

            val output = if(opInfo.name in nonConformantOps) {
                null
            } else {
                factory.create(ii.dimension(0), ii.dimension(1))
            }

            arguments.add(output)
        }

        arguments.add(args.get(0).image)
        arguments.add(args.get(1).image)
        arguments.addAll(parameters)

        val result = ImageJOpsOperationLoader.ops.run(opInfo.name, *(arguments.toTypedArray()))

        val duration = System.nanoTime() - start
        ImageJOpsOperationLoader.printlnMaybe("${Thread.currentThread().name}: ${opInfo.name} took ${duration / 10e5}ms")
        try {
            if (ImageJOpsOperationLoader.debugOps) {
//                    ImageJFunctions.showFloat(result as RandomAccessibleInterval<FloatType>, opInfo.name)
//                    ui.show(opInfo.name, result)
                val filename = "${Thread.currentThread().name}-${System.currentTimeMillis()}-binary-${opInfo.name}.tiff"
                println("Saving result $result to $filename via ${ImageJOpsOperationLoader.io.getSaver(result, filename)}")
                val ds = DefaultDataset(ImageJOpsOperationLoader.context, ImgPlus.wrap(result as Img<UnsignedByteType>))
                ImageJOpsOperationLoader.io.save(ds, filename)
            }
        } catch (e: Exception) {
            System.err.println("Exception occured while showing debug image: ${e.cause}")
            e.printStackTrace()
        }

        Image.ImgLib2Image(result as IterableInterval<*>) as T
    } catch (e: Exception) {
        ImageJOpsOperationLoader.printlnMaybe("${Thread.currentThread().name}: Execution of binary ${opInfo.name} failed, returning RHS input image.")
        ImageJOpsOperationLoader.printlnMaybe("${Thread.currentThread().name}: Parameters were: ${parameters.joinToString(",")}")
        if(!ImageJOpsOperationLoader.silent) {
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
        return "r[$destination] = ops.run(\"$representation\", r[${operands[0]}], r[${operands[1]}] ${ImageJOpsOperationLoader.parametersToCode(parameters)})"
    }

    override val information = ModuleInformation(
            description = ImageJOpsOperationLoader.ops.help(opInfo.name)
    )

    override fun mutateParameters(): BinaryOpsOperation<T> {
//            println("Old parameters: ${parameters.joinToString(", ")}")
        val newParameters = ImageJOpsOperationLoader.augmentParameters(opInfo)
//            println("New parameters: ${newParameters.joinToString(", ")}")
        return BinaryOpsOperation<T>(opInfo, newParameters, requiresInOut)
    }

    override fun toString(): String {
        return representation
    }

    companion object {
        val nonConformantOps = listOf("math.divide", "math.subtract", "math.add", "math.multiply")
    }
}
