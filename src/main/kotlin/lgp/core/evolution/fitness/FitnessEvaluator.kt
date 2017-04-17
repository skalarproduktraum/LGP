package lgp.core.evolution.fitness

import lgp.core.environment.Environment
import lgp.core.evolution.population.Program

/**
 * An evaluation of a program on a set of fitness cases.
 *
 * @param fitness The fitness of the program as determined by the fitness function on the cases given by a fitness context.
 * @param result The final result of the program tested.
 */
data class Evaluation<out T>(val fitness: Double, val result: T)

/**
 * Provides a way to evaluate the fitness of a program.
 *
 * @param T The type of the program being evaluated.
 */
class FitnessEvaluator<T> {
    /**
     * Performs an evaluation on the program given with the specified environment by building a fitness context.
     *
     * An environment is given as a parameter as it defines the fitness cases
     *
     * @param program The program to determine an evaluation for.
     * @param environment An environment that was used to build the program given.
     * @returns An evaluation of the program.
     */
    // TODO: Could environment be an instance variable?
    fun evaluate(program: Program<T>, environment: Environment<T>): Evaluation<T> {
        // Build a fitness context for this program
        val context = FitnessContext<T>(
                program = program,
                fitnessCases = environment.dataset.instances, // We want to test the program on the dataset given
                fitnessFunction = environment.fitnessFunction // User specified fitness function
        )

        return Evaluation(
                fitness = context.fitness(),
                // TODO: This should be a parameter
                // TODO: How to handle multiple program outputs (is this required?)
                result = program.registers.read(0)
        )
    }
}