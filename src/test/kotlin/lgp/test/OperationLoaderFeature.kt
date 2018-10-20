package lgp.test

import lgp.core.environment.ComponentLoadException
import lgp.core.environment.operations.DefaultOperationLoader
import lgp.core.environment.operations.InvalidOperationSpecificationException
import lgp.core.program.instructions.Operation
import lgp.lib.operations.Addition
import lgp.lib.operations.Subtraction
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.gherkin.Feature

object OperationLoaderFeature : Spek({
    Feature("DefaultOperationLoader") {
        Scenario("Load double-typed operations from valid list") {
            lateinit var operationsRaw: List<String>
            var operationsLoaded: List<Operation<Double>>? = null
            lateinit var operationLoader: DefaultOperationLoader<Double>

            Given("The list of operations [\"lgp.lib.operations.Addition\", \"lgp.lib.operations.Subtraction\"]") {
                operationsRaw = listOf("lgp.lib.operations.Addition", "lgp.lib.operations.Subtraction")
            }

            And("A DefaultOperationLoader for the operations list") {
                operationLoader = DefaultOperationLoader(operationsRaw)
            }

            When("The operations are loaded") {
                operationsLoaded = operationLoader.load()
            }

            Then("The operations are loaded successfully") {
                assert(
                    operationsLoaded != null &&
                    operationsLoaded!!.size == operationsRaw.size
                ) { "Loaded operations is null or the number of operations loaded is not correct" }

                val firstOperation = operationsLoaded!!.first()
                val lastOperation = operationsLoaded!!.last()

                assert(
                    firstOperation is Addition && lastOperation is Subtraction
                ) { "One of the operations loaded (or both) is not the correct type" }
            }
        }

        Scenario("Load double-typed operations from invalid list") {
            lateinit var operationsRaw: List<String>
            var operationsLoaded: List<Operation<Double>>? = null
            lateinit var operationLoader: DefaultOperationLoader<Double>
            var exception: Exception? = null

            Given("The list of operations [\"lgp.lib.operations.NotValid1\", \"lgp.lib.operations.NotValid2\"]") {
                operationsRaw = listOf("lgp.lib.operations.NotValid1", "lgp.lib.operations.NotValid2")
            }

            And("A DefaultOperationLoader for the operations list") {
                operationLoader = DefaultOperationLoader(operationsRaw)
            }

            When("The operations are loaded") {
                try {
                    operationsLoaded = operationLoader.load()
                } catch (ex: Exception) {
                    exception = ex
                }
            }

            Then("The operations are not loaded successfully") {
                assert(operationsLoaded == null) { "Loaded operations is not null" }
                assert(exception != null) { "Exception is null"}
                assert(exception is ComponentLoadException) { "Exception is not of correct type" }
                assert(exception!!.cause is InvalidOperationSpecificationException) { "Inner exception is not of correct type "}
            }
        }
    }
})