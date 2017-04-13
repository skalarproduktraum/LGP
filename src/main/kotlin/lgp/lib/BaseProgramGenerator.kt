package lgp.lib

import lgp.core.environment.Environment
import lgp.core.evolution.population.Program
import lgp.core.evolution.population.ProgramGenerator
import lgp.core.modules.ModuleInformation
import lgp.lib.BaseInstructionGenerator
import lgp.lib.BaseProgram
import java.util.*

/**
 * @suppress
 */
class BaseProgramGenerator<T>(environment: Environment<T>)
    : ProgramGenerator<T>(environment, instructionGenerator = BaseInstructionGenerator(environment)) {

    private val rg = Random()

    override fun generateProgram(): Program<T> {
        val length = this.rg.randint(this.environment.config.initialMinimumProgramLength,
                                     this.environment.config.initialMaximumProgramLength)

        val instructions = this.instructionGenerator.next().take(length)

        val program = BaseProgram(instructions, this.instructionGenerator.registers)

        return program
    }

    override val information: ModuleInformation = object : ModuleInformation {
        override val description: String
            get() = "A simple program generator."
    }

}

// A random integer between a and b.
// a <= b
fun Random.randint(a: Int, b: Int): Int {
    assert(a <= b)

    return this.nextInt((b - a) + 1) + a
}