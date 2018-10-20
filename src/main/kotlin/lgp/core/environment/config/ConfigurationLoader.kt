package lgp.core.environment.config

import lgp.core.environment.ComponentLoader
import java.lang.Exception

/**
 * An extended [ComponentLoader] that is responsible for loading [Configuration] instances.
 *
 * The method in which the configuration is loaded is to be defined through an implementation.
 *
 * @see [Configuration]
 */
interface ConfigurationLoader : ComponentLoader<Configuration>

/**
 * Exception raised when configuration could not be loaded by a [ConfigurationLoader].
 */
class ConfigurationLoadException(message: String, exception: Exception) : Exception(message, exception)