Test task for ELMA intent classification and flask deployment

In **notebook/ELMO-tesk-task** I made EDA and Model with hyperparameters tuning
## Commands
```.bash
#Before everything
pip install -r requirement.txt

#Run app.py that contains all the required for Flask and manage APIs to deploy intent classification
python app.py

#Run intent classification on Docker
docker compose up --build
