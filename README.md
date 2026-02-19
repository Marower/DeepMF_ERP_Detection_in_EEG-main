Steps to repeat the analysis from paper:

1. Preparation
   - Download EEGLAB from: https://sccn.ucsd.edu/eeglab/download.php
In my case, it was placed in the project folder under eeglab2022.1. Otherwise, modify the EEGLAB path in the MATLAB scripts accordingly.
   - Install the BioSig plugin in EEGLAB (via the EEGLAB GUI) to enable loading .bdf files.
   - Download the dataset from:
https://openneuro.org/datasets/ds005284/versions/1.0.0/download
Place it in the repository folder or modify the scripts to point to the correct location.

In MatLab:

2. Run PrepareDataset_ds005284.m
This script loads the files, performs preprocessing, converts the data into a cell array, and saves the result as a .mat file.

3. For visual inspection of evoked potentials (EP), run:
plotERPforChannels
plotERPforSubjects
The results are stored in the EvokedPotentialPlots folder. The results of the visual inspection are stored in a spreadsheet.

4. Export data for training using:
exportDataForEncoderDecoder.m

5. Build channel filter templates using:
exportChannelsTemplates.m
The results are saved in channels_templates.mat.


In python:

6. Run trainEncoderDecoder.py to train encoder-decoders. 
Set useTemplates flag if you want to use templates to initialize model. Trained model are stord in repositorium (TemplateInitializedEncoderDecoder.pth and RandomInitializedEncoderDecoder.pth)

7. Run trainERPDetector to train models for leave-one-out validation. The encoder part of the detector model is initialized using the encoder-decoder from step 6. The use_templates flag determines whether the initialized encoder-decoder is used.

8. The Jupyter notebook testERPDetector.ipynb contains basic tests to verify the correct operation of the models.

9. Script leaceOneOutValidiation perform LOO for all trained networks, results are saved in LOO_results.mat. Use plotF1scoreForLOOO.m for visualization.