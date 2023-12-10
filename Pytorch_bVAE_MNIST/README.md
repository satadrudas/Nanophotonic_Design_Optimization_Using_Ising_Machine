This directory contains the implemententation of bVAE in on MNIST dataset using pytorch framework. It appears that the weight of he KL divergent have to be carefully chosen to get a deccent reconstruction and generation. Try with taking very large value for `beta` and then very small value and then gradually you'll get some idea on which `beta` value to use. <br/>

TODO: Do hyperparameter optimization of `beta` in the fly when training the network itself.<br/><br/>
Note: When loading a pretrained model to continue training, make sure that the lambda (the temperature) is where we left it at...
