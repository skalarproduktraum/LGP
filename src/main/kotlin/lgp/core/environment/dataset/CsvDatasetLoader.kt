package lgp.core.environment.dataset

import com.opencsv.CSVReader
import lgp.core.environment.ComponentLoaderBuilder
import lgp.core.modules.ModuleInformation
import java.io.FileReader
import java.io.Reader

// These type aliases help to make the code look nicer.
typealias Header = Array<String>
typealias Row = Array<String>

/**
 *
 */
class InvalidCsvFileException(message: String) : Exception(message)

/**
 * Loads a collection of samples and their target values from a CSV file.
 *
 * @param T Type of the features in the samples.
 * @property reader A reader that will provide the contents of a CSV file.
 * @property featureParseFunction A function to parse the features of each row in the CSV file.
 * @property targetParseFunction A function to parse the target of each row in the CSV file.
 */
class CsvDatasetLoader<out T> constructor(
        val reader: Reader,
        val featureParseFunction: (Header, Row) -> Sample<T>,
        val targetParseFunction: (Header, Row) -> T
) : DatasetLoader<T> {

    private constructor(builder: Builder<T>)
            : this(builder.reader, builder.featureParseFunction, builder.targetParseFunction)

    private val cachedDataset: Dataset<T>? = null

    /**
     * Builds an instance of [CsvDatasetLoader].
     *
     * @param U the type that the [CsvDatasetLoader] will load features as.
     */
    class Builder<U> : ComponentLoaderBuilder<CsvDatasetLoader<U>> {

        lateinit var reader: Reader
        lateinit var featureParseFunction: (Header, Row) -> Sample<U>
        lateinit var targetParseFunction: (Header, Row) -> U

        /**
         * Sets the filename of the CSV file to load the data set from.
         *
         * A reader will be automatically created for the file with the name given.
         */
        fun filename(name: String): Builder<U> {
            this.reader = FileReader(name)

            return this
        }

        /**
         * Sets the reader that provides a CSV files contents.
         */
        fun reader(reader: Reader): Builder<U> {
            this.reader = reader

            return this
        }

        /**
         * Sets the function to use when parsing features from the data set file.
         */
        fun featureParseFunction(function: (Header, Row) -> Sample<U>): Builder<U> {
            this.featureParseFunction = function

            return this
        }

        /**
         * Sets the function to use when parsing target values from the data set file.
         */
        fun targetParseFunction(function: (Header, Row) -> U): Builder<U> {
            this.targetParseFunction = function

            return this
        }

        /**
         * Builds the instance with the given configuration information.
         */
        override fun build(): CsvDatasetLoader<U> {
            return CsvDatasetLoader(this)
        }
    }

    /**
     * Loads a data set from the CSV file specified when the loader was built.
     *
     * @throws [java.io.IOException] when the file given does not exist.
     * @returns a data set containing values parsed appropriately.
     */
    override fun load(): Dataset<T> {
        // Prevent loading the data set again.
        if (this.cachedDataset != null)
            return this.cachedDataset

        val reader = CSVReader(this.reader)
        val lines: MutableList<Array<String>> = reader.readAll()

        // Make sure there is data before we continue. There should be at least two lines in the file
        // (a header and one row of data). This check will let through a file with 2 data rows, but
        // there is not much that can be done -- plus things will probably break down later on...
        if (lines.size < 2)
            throw InvalidCsvFileException("CSV file should have a header row and one or more data rows.")

        // Assumes the header is in the first row (a reasonable assumption with CSV files).
        val header = lines.removeAt(0)

        // Parse features and target values individually.
        val features = lines.map { line ->
            this.featureParseFunction(header, line)
        }

        val targets = lines.map { line ->
            this.targetParseFunction(header, line)
        }

        return Dataset(features, targets)
    }

    override val information = ModuleInformation(
        description = "A loader than can load data sets from CSV files."
    )
}
object ParsingFunctions {

    fun indexedDoubleFeatureParsingFunction(featureIndices: IntRange): (Header, Row) -> Sample<Double> {
        return { header: Header, row: Row ->
            val features = row.zip(header)
                    .slice(featureIndices)
                    .map { (featureValue, featureName) ->

                        Feature(
                                name = featureName,
                                value = featureValue.toDouble()
                        )
                    }

            Sample(features)
        }
    }

    fun indexedDoubleTargetParsingFunction(targetIndex: Int): (Header, Row) -> Double {
        return { header: Header, row: Row ->
            row[targetIndex].toDouble()
        }
    }
}