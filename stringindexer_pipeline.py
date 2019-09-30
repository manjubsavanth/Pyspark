def string_to_index(self, input_cols):
        """
        Maps a string column of labels to an ML column of label indices. If the input column is
        numeric, we cast it to string and index the string values.
        :param input_cols: Columns to be indexed.
        :return: Dataframe with indexed columns.
        """

        # Check if columns argument must be a string or list datatype:
        self._assert_type_str_or_list(input_cols, "input_cols")

        if isinstance(input_cols, str):
            input_cols = [input_cols]

        from pyspark.ml import Pipeline
        from pyspark.ml.feature import StringIndexer

        indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").fit(self._df) for column in
                    list(set(input_cols))]

        pipeline = Pipeline(stages=indexers)
        self._df = pipeline.fit(self._df).transform(self._df)

        return self 
