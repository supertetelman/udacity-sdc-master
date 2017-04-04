python feature_extraction.py --run_old --download --training_file data/cifar10-train.p --validation_file data/cifar10-validation.p

python feature_extraction.py --training_file data/vgg_traffic_100_bottleneck_features_train.p --validation_file data/vgg_traffic_bottleneck_features_validation.p
python feature_extraction.py --training_file data/resnet_traffic_100_bottleneck_features_train.p --validation_file data/resnet_traffic_bottleneck_features_validation.p
python feature_extraction.py --training_file data/inception_traffic_100_bottleneck_features_train.p --validation_file data/inception_traffic_bottleneck_features_validation.p

python feature_extraction.py --training_file data/vgg_cifar10_100_bottleneck_features_train.p --validation_file data/vgg_cifar10_bottleneck_features_validation.p
python feature_extraction.py --training_file data/resnet_cifar10_100_bottleneck_features_train.p --validation_file data/resnet_cifar10_bottleneck_features_validation.p
python feature_extraction.py --training_file data/inception_cifar10_100_bottleneck_features_train.p --validation_file data/inception_cifar10_bottleneck_features_validation.p
