To get text interface for the project run user.py, to run make sure all the latest version of all the required libraries are installed.
Some of the features may require the VIST dataset which is not included in this zip. The file structure of the dataset should be as follows with year3Project and DataSets being in the same folder:

|year3Project
	| (contents of year3Project)
|DataSets
	|VIST
		|dii
			|test.description-in-isolation.json
			|train.description-in-isolation.json
			|val.description-in-isolation.json
		|miniTraining
			|train
				|(images in dataset)
		|miniValidation
			|val
				|(images in dataset)
		|sis
			|test.story-in-sequence.json
			|train.story-in-sequence.json
			|val.story-in-sequence.json
		|testing
			|test
				|(images in dataset)
		|training
			|train
				|(images in dataset)
		|validation
			|val
				|(images in dataset)

It should also be noted that the datasets for training the image encoder have been removed from the models folder to save space in zip folder, however this shouldn't effect usability.