package lgp.examples

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
import net.imglib2.RandomAccessibleInterval
import net.imglib2.img.cell.CellImgFactory
import net.imglib2.type.numeric.real.FloatType
import net.imglib2.view.IterableRandomAccessibleInterval
import net.imglib2.view.Views
import java.lang.RuntimeException

/*
 * An example of setting up an environment to use LGP to find programs for the function `x^2 + 2x + 2`.
 *
 * This example serves as a good way to learn how to use the system and to ensure that everything
 * is working correctly, as some percentage of the time, perfect individuals should be found.
 */

// A solution for this problem consists of the problem's name and a result from
// running the problem with a `Trainer` impl.
data class SpotFinderSolution(
    override val problem: String,
    val result: TrainingResult<RandomAccessibleInterval<*>, Outputs.Single<RandomAccessibleInterval<*>>>
) : Solution<RandomAccessibleInterval<*>>

// Define the problem and the necessary components to solve it.
class SpotFinderProblem: Problem<RandomAccessibleInterval<*>, Outputs.Single<RandomAccessibleInterval<*>>>() {
    override val name = "Simple Quadratic."

    override val description = Description("f(x) = x^2 + 2x + 2\n\trange = [-10:10:0.5]")

    override val configLoader = object : ConfigurationLoader {
        override val information = ModuleInformation("Overrides default configuration for this problem.")

        override fun load(): Configuration {
            val config = Configuration()

            config.initialMinimumProgramLength = 10
            config.initialMaximumProgramLength = 30
            config.minimumProgramLength = 10
            config.maximumProgramLength = 200
            config.operations = listOf(
                "lgp.lib.operations.Addition",
                "lgp.lib.operations.Subtraction",
                "lgp.lib.operations.Multiplication"
            )
            config.constantsRate = 0.5
            config.constants = listOf("0.0", "1.0", "2.0")
            config.numCalculationRegisters = 4
            config.populationSize = 500
            config.generations = 1000
            config.numFeatures = 1
            config.microMutationRate = 0.4
            config.macroMutationRate = 0.6
            config.numOffspring = 10

            return config
        }
    }

    private val config = this.configLoader.load()

    override val constantLoader = GenericConstantLoader(
        constants = config.constants,
        parseFunction = { s ->
            val f = CellImgFactory(FloatType(), 2)
            val img = f.create(512, 512)
            val rai = Views.interval(img, longArrayOf(0, 0), longArrayOf(512, 512))
            val cursor = rai.cursor()
            while(cursor.hasNext()) {
                cursor.fwd()
                cursor.get().set(s.toFloat())
            }

            rai as RandomAccessibleInterval<*>
        }
    )

    val inputFiles = arrayListOf<String>()
    val outputFiles = arrayListOf<String>()

    val datasetLoader = object : DatasetLoader<RandomAccessibleInterval<*>> {
        override val information = ModuleInformation("Generates samples in the range [-10:10:0.5].")

        override fun load(): Dataset<RandomAccessibleInterval<*>> {
            val inputs = inputFiles.map { filename ->
                val img = IO.openImgs(filename).get(0)
                Sample(listOf(Feature(name = "image", value = img as RandomAccessibleInterval<*>)))
            }

            val outputs = outputFiles.map { filename ->
                val img = IO.openImgs(filename).get(0)
                Targets.Single(img as RandomAccessibleInterval<*>)
            }

            return Dataset(
                inputs.toList(),
                outputs.toList()
            )
        }
    }

    override val operationLoader = ImageJOpsOperationLoader<RandomAccessibleInterval<*>>(
        typeFilter = RandomAccessibleInterval::class.java,
        opsFilter= config.operations
    )
    val defaultImage: RandomAccessibleInterval<*>
    val whiteImage: RandomAccessibleInterval<*>

    init {
        val factory = CellImgFactory(FloatType(), 2)
        val img = factory.create(512, 512)

        defaultImage = Views.interval(img, longArrayOf(0, 0), longArrayOf(512, 512))

        val whiteImg = factory.create(512, 512)

        whiteImage = Views.interval(whiteImg, longArrayOf(0, 0), longArrayOf(512, 512))
        val cursor = whiteImg.cursor()
        while(cursor.hasNext()) {
            cursor.fwd()
            cursor.get().set(1.0f)
        }
    }

    override val defaultValueProvider = DefaultValueProviders.constantValueProvider(defaultImage)

    override val fitnessFunctionProvider = {
        val ff: SingleOutputFitnessFunction<RandomAccessibleInterval<*>> = object : SingleOutputFitnessFunction<RandomAccessibleInterval<*>>() {

            override fun fitness(outputs: List<Outputs.Single<RandomAccessibleInterval<*>>>, cases: List<FitnessCase<RandomAccessibleInterval<*>>>): Double {
                val fitness = cases.zip(outputs).map { (case, actual) ->
                    val raiExpected = (case.target as Targets.Single).value
                    val raiActual = actual.value

                    if(raiExpected !is IterableRandomAccessibleInterval || raiActual !is IterableRandomAccessibleInterval) {
                        throw RuntimeException("One of the RAIs is not iterable")
                    }

                    val cursorExpected = raiExpected.cursor()
                    val cursorActual = raiActual.cursor()

                    var difference = 0.0f
                    while(cursorActual.hasNext() && cursorExpected.hasNext()) {
                        cursorActual.fwd()
                        cursorExpected.fwd()

                        difference = (cursorActual.get() as Float) - (cursorExpected.get() as Float)
                    }

                    difference
                }.sum()

                return when {
                    fitness.isFinite() -> ((1.0 / cases.size.toDouble()) * fitness)
                    else               -> FitnessFunctions.UNDEFINED_FITNESS
                }
            }
        }

        ff
    }

    override val registeredModules = ModuleContainer<RandomAccessibleInterval<*>, Outputs.Single<RandomAccessibleInterval<*>>>(
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
                TournamentSelection(environment, tournamentSize = 2)
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
                    constantMutationFunc = ConstantMutationFunctions.identity<RandomAccessibleInterval<*>>()
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
            ResultAggregators.InMemoryResultAggregator<RandomAccessibleInterval<*>>()
        )

        this.environment.registerModules(this.registeredModules)
    }

    override fun initialiseModel() {
        this.model = Models.SteadyState(this.environment)
    }

    override fun solve(): SpotFinderSolution {
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

            val runner = DistributedTrainer(environment, model, runs = 2)

            return runBlocking {
                val job = runner.trainAsync(
                    this@SpotFinderProblem.datasetLoader.load()
                )

                job.subscribeToUpdates { println("training progress = ${it.progress}") }

                val result = job.result()

                SpotFinderSolution(this@SpotFinderProblem.name, result)
            }

        } catch (ex: UninitializedPropertyAccessException) {
            // The initialisation routines haven't been run.
            throw ProblemNotInitialisedException(
                "The initialisation routines for this problem must be run before it can be solved."
            )
        }
    }
}

class SpotFinder {
    companion object Main {
        @JvmStatic fun main(args: Array<String>) {
            // Create a new problem instance, initialise it, and then solve it.
            val problem = SpotFinderProblem()
            problem.initialiseEnvironment()
            problem.initialiseModel()
            val solution = problem.solve()
            val simplifier = BaseProgramSimplifier<RandomAccessibleInterval<*>, Outputs.Single<RandomAccessibleInterval<*>>>()

            println("Results:")

            solution.result.evaluations.forEachIndexed { run, res ->
                println("Run ${run + 1} (best fitness = ${res.best.fitness})")
                println(simplifier.simplify(res.best as BaseProgram<RandomAccessibleInterval<*>, Outputs.Single<RandomAccessibleInterval<*>>>))

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
