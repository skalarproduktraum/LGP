package lgp.test

import lgp.core.environment.config.Configuration
import lgp.core.environment.config.JsonConfigurationLoader
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.gherkin.Feature

object ConfigurationLoaderFeature : Spek({
    val testConfigurationFilePath = this.javaClass.classLoader.getResource("test-configuration.json").file
    val expectedConfiguration = Configuration().apply {
        initialMinimumProgramLength = 20
        initialMaximumProgramLength = 40
        minimumProgramLength = 20
        maximumProgramLength = 400
        operations = listOf(
            "lgp.lib.operations.Addition",
            "lgp.lib.operations.Subtraction"
        )
        constantsRate = 0.2
        constants = listOf(
            "0.0", "0.1", "1.0"
        )
        numCalculationRegisters = 5
        populationSize = 500
        numFeatures = 2
        crossoverRate = 0.2
        microMutationRate = 0.2
        macroMutationRate = 0.8
        generations = 100
        numOffspring = 10
        branchInitialisationRate = 0.1
        stoppingCriterion = 0.0001
        numberOfRuns = 2
    }

    Feature("JsonConfigurationLoader") {
        Scenario("load configuration") {
            lateinit var configurationLoader: JsonConfigurationLoader
            var configuration: Configuration? = null

            Given("A JsonConfigurationLoader for the file test-configuration.json") {
                configurationLoader = JsonConfigurationLoader(testConfigurationFilePath)
            }

            When("The configuration is loaded") {
                configuration = configurationLoader.load()
            }

            Then("The configuration should be loaded successfully") {
                assert(configuration != null) { "Configuration is null" }
            }

            And("The configuration is valid") {
                val validity = configuration!!.isValid()

                assert(validity.isValid) { "Configuration is not valid" }
            }

            And("The configuration is loaded correctly") {
                assert(configuration == expectedConfiguration) { "Configuration loaded does not match expected"}
            }
        }
    }
})