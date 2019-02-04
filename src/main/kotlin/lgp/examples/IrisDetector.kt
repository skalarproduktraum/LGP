package lgp.examples

import io.scif.SCIFIOService
import io.scif.img.IO
import kotlinx.coroutines.runBlocking
import lgp.core.environment.*
import lgp.core.environment.config.Configuration
import lgp.core.environment.config.ConfigurationLoader
import lgp.core.environment.constants.GenericConstantLoader
import lgp.core.environment.dataset.*
import lgp.core.environment.operations.ImageJOpsOperationLoader
import lgp.core.evolution.*
import lgp.core.evolution.fitness.*
import lgp.core.evolution.model.Models
import lgp.core.evolution.operators.*
import lgp.core.evolution.training.DistributedTrainer
import lgp.core.evolution.training.TrainingResult
import lgp.core.modules.ModuleInformation
import lgp.core.program.Outputs
import lgp.lib.*
import net.imagej.DefaultDataset
import net.imagej.ImageJService
import net.imagej.ImgPlus
import net.imagej.ops.OpService
import net.imglib2.IterableInterval
import net.imglib2.RandomAccessibleInterval
import net.imglib2.img.Img
import net.imglib2.img.array.ArrayImgFactory
import net.imglib2.type.numeric.RealType
import net.imglib2.type.numeric.real.DoubleType
import net.imglib2.type.numeric.real.FloatType
import net.imglib2.view.Views
import org.scijava.Context
import org.scijava.io.IOService
import org.scijava.service.SciJavaService
import org.scijava.thread.ThreadService
import org.scijava.ui.UIService
import java.io.File
import java.io.FileOutputStream
import java.io.PrintStream
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.absoluteValue
import kotlin.math.sqrt

/*
 * An example of setting up an environment to use LGP to find programs for the function `x^2 + 2x + 2`.
 *
 * This example serves as a good way to learn how to use the system and to ensure that everything
 * is working correctly, as some percentage of the time, perfect individuals should be found.
 */

// A solution for this problem consists of the problem's name and a result from
// running the problem with a `Trainer` impl.
data class IrisDetectorSolution(
    override val problem: String,
    val result: TrainingResult<IterableInterval<*>, Outputs.Single<IterableInterval<*>>>
) : Solution<IterableInterval<*>>

// Define the problem and the necessary components to solve it.
class IrisDetectorProblem: Problem<IterableInterval<*>, Outputs.Single<IterableInterval<*>>>() {
    override val name = "Iris Detection"

    override val description = Description("f(x) = x^2 + 2x + 2\n\trange = [-10:10:0.5]")

    override val configLoader = object : ConfigurationLoader {
        override val information = ModuleInformation("Overrides default configuration for this problem.")

        override fun load(): Configuration {
            val config = Configuration()

            config.initialMinimumProgramLength = 2
            config.initialMaximumProgramLength = 6
            config.minimumProgramLength = 10
            config.maximumProgramLength = 50
            config.operations = listOf(
                "lgp.lib.operations.Addition",
                "lgp.lib.operations.Subtraction",
                "lgp.lib.operations.Multiplication"
            )
            config.constantsRate = 0.5
            config.constants = listOf("0.0", "1.0", "2.0")
            config.numCalculationRegisters = 4
            config.populationSize = 100
            config.generations = 1000
            config.numFeatures = 1
            config.microMutationRate = 0.7
            config.crossoverRate = 0.75
            config.macroMutationRate = 0.7
            config.numOffspring = 5

            return config
        }
    }

    private val startTime = SimpleDateFormat("yyyy-MM-dd_HH.mm.ss").format(Date())

    enum class ImageMetrics {
        TED,
        MCC,
        TEDandMCC,
        AbsoluteDifferences,
    }

    private val config = this.configLoader.load()
    private val fitnessMetric = ImageMetrics.MCC

    val imageWidth = 320L
    val imageHeight = 240L

    override val constantLoader = GenericConstantLoader(
        constants = config.constants,
        parseFunction = { s ->
            val f = ArrayImgFactory(FloatType())
            val img = f.create(imageWidth, imageHeight)
            val rai = Views.interval(img, longArrayOf(0, 0), longArrayOf(imageWidth- 1, imageHeight - 1))
            val cursor = rai.cursor()
            while(cursor.hasNext()) {
                cursor.fwd()
                cursor.get().set(s.toFloat())
            }

            rai as IterableInterval<*>
        }
    )

    val inputDirectory: String = System.getProperty("IrisDataDirectory", "IrisProject")
    val maxDirectories = 1

    val inputFiles = (1..maxDirectories)
        .map { "$inputDirectory/iitd/${String.format("%03d", it)}/"}
        .map {
            val d = File(it)
            d.listFiles().toList()
        }.flatten()

    val datasetLoader = object : DatasetLoader<IterableInterval<*>> {
        override val information = ModuleInformation("Generates samples in the range [-10:10:0.5].")

        override fun load(): Dataset<IterableInterval<*>> {
            val opService = ImageJOpsOperationLoader.ops
            val factory = ArrayImgFactory(FloatType())

            println("Loading input images ...")
            val inputs = inputFiles.map { filename ->
                println("Loading input file $filename")
                val img = io.open(filename.toString()) as Img<*>
                val floatImg = opService.run("convert.float32", img) as RandomAccessibleInterval<*>

                val final = Views.hyperSlice(floatImg, 2, 0)

//                ImageJFunctions.showFloat(final as RandomAccessibleInterval<FloatType>, filename.toString())

                Sample(listOf(Feature(name = "image", value = final as IterableInterval<*>)))
            }

            println("Loading ground truth masks ...")
            val outputs = inputFiles.mapIndexed { i, filename ->
                // convert the positions from the CSV file into a binary image
                val id = filename.name.substringBeforeLast("_").toInt()
                val AB = if(id < 6) { "A" } else { "B" }
                val maskFileName = "OperatorA_${filename.parent.substringAfterLast("/")}-${AB}_${String.format("%02d", id)}.tiff"
                val f = "$inputDirectory/IRISSEG-EP-Masks/masks/iitd/$maskFileName"
                val img = io.open(f) as Img<*>

                val floatImg = opService.run("convert.float32", img)
//                ImageJFunctions.show(floatImg as RandomAccessibleInterval<FloatType>, maskFileName)
                Targets.Single(floatImg as IterableInterval<*>)
            }

            return Dataset(
                inputs.toList(),
                outputs.toList()
            )
        }
    }

    override val operationLoader = ImageJOpsOperationLoader<IterableInterval<*>>(
        typeFilter = IterableInterval::class.java,
        opsFilter= config.operations
    )
    val defaultImage: IterableInterval<*>
    val whiteImage: IterableInterval<*>

    init {
        val factory = ArrayImgFactory(FloatType())
        val img = factory.create(imageWidth, imageHeight)

        defaultImage = Views.interval(img, longArrayOf(0, 0), longArrayOf(imageWidth-1, imageHeight-1))

        val whiteImg = factory.create(imageWidth, imageHeight)

        whiteImage = Views.interval(whiteImg, longArrayOf(0, 0), longArrayOf(imageWidth-1, imageHeight-1))
        val cursor = whiteImg.cursor()
        while(cursor.hasNext()) {
            cursor.fwd()
            cursor.get().set(1.0f)
        }
    }

    override val defaultValueProvider = DefaultValueProviders.constantValueProvider(defaultImage)

    val cache = HashMap<Any, Any>()

    fun intervalToFile(ii: IterableInterval<*>, filename: String) {
        val ds = DefaultDataset(context, ImgPlus.wrap(ii as Img<RealType<*>>))

        println("Saving $ii to $filename")
        io.save(ds, filename)
    }

    override val fitnessFunctionProvider = {
        val ff: SingleOutputFitnessFunction<IterableInterval<*>> = object : SingleOutputFitnessFunction<IterableInterval<*>>() {

            override fun fitness(outputs: List<Outputs.Single<IterableInterval<*>>>, cases: List<FitnessCase<IterableInterval<*>>>): Double {
                val fitnessAbsoluteDifferences = {
                    cases.zip(outputs).map { (case, actual) ->
                        val raiExpected = (case.target as Targets.Single).value
                        val raiActual = actual.value


                        val cursorExpected = Views.iterable(raiExpected as RandomAccessibleInterval<*>).localizingCursor()
                        val cursorActual = Views.iterable(raiActual as RandomAccessibleInterval<*>).localizingCursor()

                        var difference = 0.0f
                        var counts = 0
                        while (cursorActual.hasNext() && cursorExpected.hasNext()) {
                            cursorActual.fwd()
                            cursorExpected.fwd()

                            difference += ((cursorActual.get() as FloatType).get() -
                                    (cursorExpected.get() as FloatType).get()).absoluteValue
                            counts++
                        }

                        difference /= counts

                        if(difference < 100) {
                            val ds = DefaultDataset(context, ImgPlus.wrap(raiExpected as Img<RealType<*>>))
                            val dsActual = DefaultDataset(context, ImgPlus.wrap(raiActual as Img<RealType<*>>))
                            val timestamp = System.currentTimeMillis()
                            io.save(dsActual, "$startTime/$timestamp-actual-fitness=$difference.tiff")
                            io.save(ds, "$startTime/$timestamp-expected.tiff")
                        }

                        difference
                    }.sum()
                }

                val fitnessTED = {
                    val factory = ArrayImgFactory(FloatType())
                    cases.zip(outputs).map { (case, actual) ->
                        val raiExpected = (case.target as Targets.Single).value
                        val raiActual = actual.value

                        val dtExpected = cache.getOrPut(raiExpected) {
                            val convertedExpected = ops.run("convert.bit", raiExpected)
                            val dtExpected = factory.create(raiExpected.dimension(0), raiExpected.dimension(1))
                            ops.run("image.distancetransform", dtExpected, convertedExpected)

                            dtExpected
                        }

                        val convertedActual = ops.run("convert.bit", raiActual)
                        val dtActual = ops.run("image.distancetransform", convertedActual)

                        val difference = ops.run("math.subtract", dtExpected, dtActual)
//                        val sum = DoubleType(0.0)
                        val sum = ops.run("stats.sum", difference) as DoubleType
//                        ops.run("stats.sum", sum, difference) as DoubleType

//                        val timestamp = System.currentTimeMillis()
//                        intervalToFile(dtExpected as IterableInterval<*>, "$startTime/$timestamp-expected.tiff")
//                        intervalToFile(dtActual as IterableInterval<*>, "$startTime/$timestamp-actual.tiff")
//                        intervalToFile(difference as IterableInterval<*> , "$startTime/$timestamp-difference.tiff")

                        sum.get().toFloat().absoluteValue/(raiActual.dimension(0)*raiActual.dimension(1))
                    }.sum()
                }

                val fitnessMatthewsCorrelationCoefficient = {
                    cases.zip(outputs).map { (case, actual) ->
                        var trueNegatives = 0L
                        var falseNegatives = 0L
                        var truePositives = 0L
                        var falsePositives = 0L

                        var prev = Float.NaN
                        var totalDifference = 0.0f
                        var totalDifferenceOriginal = 0.0f

                        val raiOriginal = (case.features.features.first()).value
                        val raiExpected = (case.target as Targets.Single).value
                        val raiActual = actual.value

                        val cursorOriginal = Views.iterable(raiOriginal as RandomAccessibleInterval<*>).localizingCursor()
                        val cursorExpected = Views.iterable(raiExpected as RandomAccessibleInterval<*>).localizingCursor()
//                        val cursorActual = Views.iterable(raiActual as RandomAccessibleInterval<*>).localizingCursor()

                        val thresholded = ops.run("threshold.maxEntropy", raiActual)
                        val converted = ops.run("convert.float32", thresholded) as IterableInterval<*>
                        val cursorActual = Views.iterable(converted as RandomAccessibleInterval<*>).localizingCursor()

                        while(cursorActual.hasNext() && cursorExpected.hasNext() && cursorOriginal.hasNext()) {
                            cursorOriginal.fwd()
                            cursorActual.fwd()
                            cursorExpected.fwd()

                            val originalValue = (cursorOriginal.get() as FloatType).get()
                            val actualValue = (cursorActual.get() as FloatType).get()
                            val expectedValue = (cursorExpected.get() as FloatType).get()

                            if(prev.isNaN()) {
                                prev = actualValue
                            }

                            totalDifference += (prev - actualValue).absoluteValue
                            totalDifferenceOriginal += (actualValue - originalValue).absoluteValue

                            prev = actualValue

                            if(expectedValue < 254.9f && actualValue < 1.0f) {
                                trueNegatives++
                            }

                            if(expectedValue > 254.9f && actualValue < 1.0f) {
                                falseNegatives++
                            }

                            if(expectedValue > 254.9f && actualValue >= 1.0f) {
                                truePositives++
                            }

                            if(expectedValue < 254.9f && actualValue >= 1.0f) {
                                falsePositives++
                            }
                        }

                        val denom = (truePositives + falsePositives) * (truePositives + falseNegatives) * (trueNegatives + falsePositives) * (trueNegatives + falseNegatives)
                        val mccDenom = if(denom == 0L) {
                            1.0
                        } else {
                            sqrt(denom.toDouble())
                        }

                        val mcc = (truePositives * trueNegatives - falsePositives * falseNegatives)/mccDenom
                        println("${Thread.currentThread().name}:MCC=$mcc, TP=$truePositives, FP=$falsePositives, TN=$trueNegatives, FN=$falseNegatives, delta=$totalDifference, delta_o=$totalDifferenceOriginal")

                        if(1.0f - mcc.toFloat().absoluteValue < 0.2f) {
                            val ds = DefaultDataset(context, ImgPlus.wrap(raiExpected as Img<RealType<*>>))
                            val dsActual = DefaultDataset(context, ImgPlus.wrap(converted as Img<RealType<*>>))
                            val timestamp = System.currentTimeMillis()
                            val filename = "$startTime/$timestamp-actual-fitness=${1.0f-mcc.toFloat().absoluteValue}.tiff"
                            println("${Thread.currentThread().name}:Saving actual to $filename")
                            io.save(dsActual, filename)
                            io.save(ds, "$startTime/$timestamp-expected.tiff")
                        }

                        if(totalDifference < 10.0f || totalDifferenceOriginal < 10.0f) {
                            1.0f
                        } else {
                            1.0f - mcc.toFloat().absoluteValue
                        }
                    }.sum()
                }

                val fitness = try {
                    when(fitnessMetric) {
                        ImageMetrics.TED -> fitnessTED.invoke()
                        ImageMetrics.MCC -> fitnessMatthewsCorrelationCoefficient.invoke()
                        ImageMetrics.TEDandMCC -> {
                            val mcc = fitnessMatthewsCorrelationCoefficient.invoke()
                            val ted = fitnessTED.invoke()

                            println("${Thread.currentThread().name}:MCC = ${mcc/cases.size}")
                            ted
                        }
                        ImageMetrics.AbsoluteDifferences -> fitnessAbsoluteDifferences.invoke()
                    }
                } catch (e: Exception) {
                    println("${Thread.currentThread().name}:Failed Fitness evaluation: ${e.toString()}")
                    Float.NEGATIVE_INFINITY
                }

                val f = when {
                    fitness.isFinite() -> fitness / cases.size.toDouble()
                    else               -> FitnessFunctions.UNDEFINED_FITNESS
                }

                println("${Thread.currentThread().name}:Fitness = $f")
                return f
            }
        }

        ff
    }

    override val registeredModules = ModuleContainer<IterableInterval<*>, Outputs.Single<IterableInterval<*>>>(
        modules = mutableMapOf(
            CoreModuleType.InstructionGenerator to { environment ->
                BaseInstructionGenerator(environment)
            },
            CoreModuleType.ProgramGenerator to { environment ->
                BaseProgramGenerator(
                    environment,
                    sentinelTrueValue = whiteImage,
                    outputRegisterIndices = listOf(0),
                    outputResolver = BaseProgramOutputResolvers.singleOutput()
                )
            },
            CoreModuleType.SelectionOperator to { environment ->
                TournamentSelection(environment, tournamentSize = 8, speciation = SpeciationSelection(2, 0.5f))
            },
            CoreModuleType.RecombinationOperator to { environment ->
                LinearCrossover(
                    environment,
                    maximumSegmentLength = 6,
                    maximumCrossoverDistance = 5,
                    maximumSegmentLengthDifference = 3
                )
            },
            CoreModuleType.MacroMutationOperator to { environment ->
                MacroMutationOperator(
                    environment,
                    insertionRate = 0.67,
                    deletionRate = 0.33
                )
            },
            CoreModuleType.MicroMutationOperator to { environment ->
                MicroMutationOperator(
                    environment,
                    registerMutationRate = 0.5,
                    operatorMutationRate = 0.5,
                    // Use identity func. since the probabilities
                    // of other micro mutations mean that we aren't
                    // modifying constants.
                    constantMutationFunc = ConstantMutationFunctions.identity<IterableInterval<*>>()
                )
            },
            CoreModuleType.FitnessContext to { environment ->
                SingleOutputFitnessContext(environment)
            }
        )
    )

    override fun initialiseEnvironment() {
        this.environment = Environment(
            this.configLoader,
            this.constantLoader,
            this.operationLoader,
            this.defaultValueProvider,
            this.fitnessFunctionProvider,
            ResultAggregators.InMemoryResultAggregator<IterableInterval<*>>()
        )

        this.environment.registerModules(this.registeredModules)
    }

    override fun initialiseModel() {
        this.model = Models.IslandMigration(this.environment,
            Models.IslandMigration.IslandMigrationOptions(4, 2, 20))
//        this.model = Models.SteadyState(this.environment)
    }

    override fun solve(): IrisDetectorSolution {
        try {
            /*
            // This is an example of training sequentially in an asynchronous manner.
            val runner = SequentialTrainer(environment, model, runs = 2)

            return runBlocking {
                val job = runner.trainAsync(
                    this@SimpleFunctionProblem.datasetLoader.load()
                )

                job.subscribeToUpdates { println("training progress = ${it.progress}%") }

                val result = job.result()

                SimpleFunctionSolution(this@SimpleFunctionProblem.name, result)
            }
            */

            val runner = DistributedTrainer(environment, model, runs = 4)

            return runBlocking {
                val job = runner.trainAsync(
                    this@IrisDetectorProblem.datasetLoader.load()
                )

                job.subscribeToUpdates { println("training progress = ${it.progress}") }

                val result = job.result()

                IrisDetectorSolution(this@IrisDetectorProblem.name, result)
            }

        } catch (ex: UninitializedPropertyAccessException) {
            // The initialisation routines haven't been run.
            throw ProblemNotInitialisedException(
                "The initialisation routines for this problem must be run before it can be solved."
            )
        }
    }

    init {
       File(startTime).mkdir()

        val logFile = File("$startTime/run.log")
        val logOutput = PrintStream(FileOutputStream(logFile, true))

        val stdout = TeeStream(System.out, logOutput)
        val stderr = TeeStream(System.err, logOutput)

        System.setOut(stdout)
        System.setErr(stderr)
    }

    companion object {
        val context = Context(
            IOService::class.java,
            UIService::class.java,
            OpService::class.java,
            SciJavaService::class.java,
            ImageJService::class.java,
            ThreadService::class.java,
            SCIFIOService::class.java
        )
        val ops = context.getService(OpService::class.java) as OpService
        val io = context.getService(IOService::class.java) as IOService
    }
}

class TeeStream(out1: PrintStream, internal var output: PrintStream) : PrintStream(out1) {
    override fun write(buf: ByteArray, off: Int, len: Int) {
        try {
            super.write(buf, off, len)
            output.write(buf, off, len)
        } catch (e: Exception) {
        }

    }

    override fun flush() {
        super.flush()
        output.flush()
    }
}

class IrisDetector {
    companion object Main {
        @JvmStatic fun main(args: Array<String>) {
            // Create a new problem instance, initialise it, and then solve it.
            val problem = IrisDetectorProblem()
            problem.initialiseEnvironment()
            problem.initialiseModel()
            println("IrisDetector: Loading images and ground truth masks from ${problem.maxDirectories} directories")
            val solution = problem.solve()
            val simplifier = BaseProgramSimplifier<IterableInterval<*>, Outputs.Single<IterableInterval<*>>>()

            println("Results:")

            solution.result.evaluations.forEachIndexed { run, res ->
                println("Run ${run + 1} (best fitness = ${res.best.fitness})")
                println(simplifier.simplify(res.best as BaseProgram<IterableInterval<*>, Outputs.Single<IterableInterval<*>>>))

                println("\nStats (last run only):\n")

                for ((k, v) in res.statistics.last().data) {
                    println("$k = $v")
                }
                println("")
            }

            val avgBestFitness = solution.result.evaluations.map { eval ->
                eval.best.fitness
            }.sum() / solution.result.evaluations.size

            println("Average best fitness: $avgBestFitness")
        }
    }
}
