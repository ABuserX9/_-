classdef TransformerAutoencoderWrapper < handle
    properties
        Model
        InputDim
        FeatureDim
        OutputDim
        Lambda
    end
    
    methods
        function obj = TransformerAutoencoderWrapper(inputDim, featureDim, outputDim, lambda)
            % Constructor to initialize the PermutationInvariantDAE model
            if count(py.sys.path, '') == 0
                insert(py.sys.path, int32(0), '');
            end
            
            % Import the Python module (assuming your Python code is in a file named PermutationInvariantDAE.py)
            % Ensure parameters are correctly passed as Python integers/floats
            inputDimPy = py.int(inputDim);
            featureDimPy = py.int(featureDim);
            outputDimPy = py.int(outputDim);
            lambdaPy = py.float(lambda);

            % Initialize the updated PermutationInvariantDAE100 model
            % Make sure the path to your Python file is correct
            obj.Model = py.TransformerAutoencoder.TransformerAutoencoder(inputDimPy, featureDimPy, outputDimPy);
            obj.InputDim = inputDim;
            obj.FeatureDim = featureDim;
            obj.OutputDim = outputDim;
            obj.Lambda = lambda;
        end
        
        function train(obj, inputData, labels, numEpochs, learningRate)
            % Convert MATLAB data to Python-compatible data
            inputDataPy = py.numpy.array(inputData);
            labelsPy = py.numpy.array(labels);
            
            % Set batch size to match the number of samples (or a suitable batch size)
            batchSize = size(inputData, 1);  % Or a smaller batch size if needed
            batchSizePy = py.int(batchSize);
            
            % Call the load_data_explicit function from utilities_PIDAE100.py
            trainLoader = py.utilities_TransformerAutoencoder.load_data_explicit(inputDataPy, labelsPy, batchSizePy);
        
            % Train the model using the train_model function from utilities_PIDAE100.py
            numEpochsPy = py.int(numEpochs);
            learningRatePy = py.float(learningRate);
            lambdaPy = py.float(obj.Lambda);

            py.utilities_TransformerAutoencoder.train_model(obj.Model, trainLoader, numEpochsPy, learningRatePy, lambdaPy);
        end
        
        function predictedLabels = predict(obj, inputData)
            % Convert MATLAB data to Python-compatible data
            inputDataPy = py.numpy.array(inputData);
            
            % Call the predict function from utilities_PIDAE100.py
            result = py.utilities_TransformerAutoencoder.predict(obj.Model, inputDataPy);
            
            % Convert the result back to a MATLAB array
            predictedLabels = double(py.array.array('d', py.numpy.nditer(result)));
            predictedLabels = reshape(predictedLabels, size(inputData)); % Reshape to match input dimensions
        end
    end
end