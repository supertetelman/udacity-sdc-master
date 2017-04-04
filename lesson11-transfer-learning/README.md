# Setup

These files accompany the two extneral repos both located in the external-resources directory.

Copy the alexnet files into the AlexNet Lab repo and and copy the other transfer-learning files into the Transfer Learning Lab repo.

`cp alexnet/* ../external-resources/CarND-Alexnet-Feature-Extraction/`
`cp transfer-learning/* ../external-resources/CarND-Transfer-Learning-Lab/`

After doing that you you will need to go into each of those two repositories and run the get_data.sh scripts.

`bash  ../external-resources/CarND-Alexnet-Feature-Extraction/get_data.sh`
`bash ../external-resources/CarND-Transfer-Learning-Lab/get_data.sh`

After that you should be good to run any of the programs you copied. In each repo running the run_models.sh shell script will duplicate the activities done during the lab.

`bash  ../external-resources/CarND-Alexnet-Feature-Extraction/run_models.sh`
`bash ../external-resources/CarND-Transfer-Learning-Lab/run_models.sh`