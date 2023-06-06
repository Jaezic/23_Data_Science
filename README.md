# Predict Korean Mountain Fire Damage
*Gachon University - 2023 Data Science Final Projects*
The goal of this project was to experience the *Big Data End-to-End Process*, which can predict the extent of damage if a fire breaks out in a forest in Korea. 

## Project Structure
```
23_Data_Science
|   config.py
|   main.py
|   main_all_eval.py
|   requirements.txt
|   
+---dataset
|   |   Dataset.py
|   |   FireDataset.csv
|   |   preprocessed.csv
|   |   
|   +---Fire
|   |       FireFacility_latlong.csv
|   |       FireStationPos.csv
|   |       FireStation_latlong.csv
|   |       FireStatistic.csv
|   |       FireStatistic_latlong.csv
|   |       MountainHeight.xlsx
|   |       MountainHeight_latlong.csv
|   |       
|   +---preprocess
|   |   |   create_dataset.py
|   |   |   preprocessing.py
|   |   |   
|   |   +---functions
|           
+---logs
|       stdout_2023-06-04_13_47_25.txt
|       
+---models
|   |   model.py
|   |   
|   +---config
|   |       ab_range.txt
|   |       dt_range.txt
|   |       gb_range.txt
|   |       knn_range.txt
|   |       lr_range.txt
|   |       rf_range.txt
|   |       voting_range.txt
|           
+---tools
|           
```
