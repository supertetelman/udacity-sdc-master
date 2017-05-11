mkdir data
cd data

curl -o non-vehicles_smallset.zip https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip 
curl -o vehicles_smallset.zip https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip
unzip non-vehicles_smallset.zip
unzip vehicles_smallset.zip