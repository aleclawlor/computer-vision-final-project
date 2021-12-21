# computer-vision-final-project
This project was created by Alec Lawlor and Anna Bolger
To use this repository download the source code. To run the main program, navigate to the downloaded folder and run main.py. Doing so will launch a Flask application on your local machine on port 5000. 

There are two datasets available to train the model. If you want to use the more accurate model which utlizes images that have been augmented from the TrashNet source, run the file model.py.
If you want to use the raw images provided by TrashNet, run the command unzip images.zip -d ./data and run the command python3 model.py

Commands:

'python main.py' : run the Flask application 
'python model.py' : train the ResNet50 model 
'python augment.py' : augment the original dataset

Anna Bolger:
•Configuration of ResNet18 Model
•Configuration of Data

Alec Lawlor:
•Configuration of web cam feed
•Data Augmentation
•Connecting model with Applicatio

