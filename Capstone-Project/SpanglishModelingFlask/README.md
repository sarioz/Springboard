# Spanglish Modeling Application and Experiments

## Use

To use the Spanglish Modeling web app, click [here](http://ec2-18-217-148-222.us-east-2.compute.amazonaws.com/).
This web app bundles a Part-of-Speech Tagger together with a Sentiment Analyzer.
This service is free to use but comes with absolutely no guarantee of anything like availability or correctness.
(This goes doubly so for this documentation!)
Please don't hit it with robots.
If you wish to run your own instances of the service, you may do so per the below.

## Deployment

In order to deploy the latest version of the web app to one of your machines from the latest official build,
first install and run the latest version of the Docker daemon.
Then you may pull the latest web app image from Docker Hub like so:

```$ docker pull sarioz/spanglish-modeling-app:latest```

After the Docker image is retrieved, to run the containerized app in the background:

```$ docker run -it -d -p 80:5000 sarioz/spanglish-modeling-app:latest```

The ```-d``` switch ensures that the container will continue running after you log out.
Make sure to have your machine accept inbound connections on port 80.
Then you can hit the app using a web browser like http://your.hostname.com (not https, which is nowadays the default).

## Web App Build and Local Run

If you wish to customize the web app, first pull the code into your local development environment.
The web app is self-contained in the ```SpanglishModelingFlask/``` directory.

In order to build the vocabularies, you need the files ```data/pos/train.conll``` and ```data/sa/train.conll```
under the ```SpanglishModelingFlask/``` directory.
These are part of the LinCE corpora.
You need a *kostenlos* [LinCE](https://ritual.uh.edu/lince/) account at to download these
files from [pos_spaeng](https://ritual.uh.edu/lince/benchmark/pos_spaeng.zip) and
[sa_spaeng](https://ritual.uh.edu/lince/benchmark/sa_spaeng.zip).
(The corpora come with a nice [paper](https://arxiv.org/abs/2005.04322) which does not require creating any account).

To run the web app locally and in development mode, use:

```$ FLASK_APP=app FLASK_ENV=development flask run```

After initialization succeeds, you can hit the app using your web browser at http://localhost:5000 (again, not https).

## Docker Build and Image Push

Once you are satisfied and want to containerize your web app, run

```$ docker build -t yourdockerhubusername/spanglish-modeling-app:latest .```

This takes approximately 6 minutes on my mid-2015 Macbook Pro (on *my* latest version).
Then you may push the built Docker image to [Docker Hub](https://hub.docker.com/) using

```$ docker image push -a yourdockerhubusername/spanglish-modeling-app```

Once this is successful, you may follow the instructions in the previous section for deployment,
making sure to change `sarioz` to `yourdockerhubusername`.

## Training and Evaluation

To replicate the
[project results](https://docs.google.com/spreadsheets/d/1PwbSxT5r1alqZVMPIM7L0D00pduHuVTRr8YP8eLYMAs/edit?usp=sharing)
or tweak the training algorithms, it is easiest to pull the entire
```Capstone-Project/``` then descend into any of the five subdirectories:
```POSTagger-RNN```, ```POSTagger-BERT```, ```SA-RNN```, ```SA-BERT```, or ```SA-Lexicon```.
The ones chosen for use in the web app are ```POSTagger-RNN``` and ```SA-RNN```.

Now you need to have not only the training sets
 ```data/pos/train.conll``` and ```data/sa/train.conll```
but also the dev sets
 ```data/pos/dev.conll``` and ```data/sa/dev.conll```, this time immediately under ```Capstone-Project```.

If you wish to use the models beside ```POSTagger-RNN``` and ```SA-RNN```,
it is up to you to customize the web app as well using the code under the respective directories.
This code layout involves some duplication but saves headache by having all 6 subdirectories (the 5 just mentioned
together with ```SpanglishModelingFlask/```) independent with one another from a 'build' perspective.

To experiment with the two ```*-BERT``` models, unpack the Multi-Language Cased BERT model 
[multi_cased_L-12_H-768_A-12](https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1)
into ```Capstone-Project``` in its own subdirectory.

While the file ```SpanglishModelingFlask/environment.yml``` contains a minimal set of dependencies for the web app,
these merely overlap with the dependencies required for experimentation and model building.
A superset of the required dependencies for experimentation and model building can be found in 
```Capstone-Project/training_environment.yml```. This file has been generated fully automatically using
```$ conda env export --no-build > training_environment.yml``` from the corresponding Conda+Pip environment
(which I have locally named ```py_3_7_tf_2_4```).
In order to automatically reconstruct this environment, make sure that [Anaconda](https://www.anaconda.com/)
is installed, and run

```$ conda env create -f=training_environment.yml -n training-environment```

Within each of the 5 subprojects, ```main_training.py``` is the entry point for training and
```main_inference.py``` is the entry point for evaluating a specific model.
