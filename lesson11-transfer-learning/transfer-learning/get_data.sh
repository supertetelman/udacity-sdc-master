mkdir data
cd data

curl -o vgg-100.zip https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b432_vgg-100/vgg-100.zip 
curl -o resnet-100.zip https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b634_resnet-100/resnet-100.zip
curl -o inception-100.zip https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b498_inception-100/inception-100.zip

tar -xvf vgg-100.zip
tar -xvf resnet-100.zip
tar -xvf inception-100.zip

cd ..

python shrink.py --network vgg --dataset traffic
python shrink.py --network vgg --dataset cifar10 
python shrink.py --network resnet --dataset traffic 
python shrink.py --network resnet --dataset cifar10 
python shrink.py --network inception --dataset traffic 
python shrink.py --network inception --dataset cifar10 
