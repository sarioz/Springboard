# Spanglish Modeling Application and Experiments

This document contains instructions organized into sections for increasing level of expertise/interest.
If something in any section is puzzling, make sure you have read the previous sections.

## Use

To use the Spanglish Modeling web app, click [here](http://ec2-18-217-148-222.us-east-2.compute.amazonaws.com/).
You may enter any tweet-like text in English, Spanish, or a mixture of the two.
This web app bundles a Part-of-Speech Tagger together with a Sentiment Analyzer.
This service is free to use but comes with absolutely no guarantee of anything like availability or correctness.
Please don't hit it with robots (if you don't know what this means, you're fine).

## Deploy

If you wish to run your own instances of the service,
you may deploy the latest version of the web app to one of your machines from the latest official build.
To do so, first install and run the [Docker daemon](https://docs.docker.com/get-docker/).
Then you may pull the latest web app image from Docker Hub like so:

```$ docker pull sarioz/spanglish-modeling-app:latest```

After the Docker image is retrieved, to run the containerized app in the background:

```$ docker run -it -d -p 80:5000 sarioz/spanglish-modeling-app:latest```

The ```-d``` switch ensures that the container will continue running after you log out.
Make sure to have your machine accept inbound connections on port 80.
Then you can hit the app using a web browser like http://your.hostname.com (not https, which is nowadays the default).

When you want to stop the container, find the associated container ID
by running

```$ docker ps```

and then run

```$ docker stop <CONTAINER ID>```


## Tweak and Run Web App Locally

If you wish to customize the web app, first pull the code into your local development environment.
The web app is self-contained in the ```SpanglishModelingFlask/``` directory.
It is simplest (though not fastest) to clone my entire Springboard repository like so:

```
$ mkdir sarioz
$ cd sarioz
$ git clone git clone https://github.com/sarioz/Springboard
$ cd Springboard/Capstone-Project/SpanglishModelingFlask
```

In order to build the vocabularies, you need the files ```data/pos/train.conll``` and ```data/sa/train.conll```
under the ```SpanglishModelingFlask/``` directory.
These are part of the LinCE corpora.
You need a *kostenlos* [LinCE](https://ritual.uh.edu/lince/) account to download these
files from [pos_spaeng](https://ritual.uh.edu/lince/benchmark/pos_spaeng.zip) and
[sa_spaeng](https://ritual.uh.edu/lince/benchmark/sa_spaeng.zip).
(The corpora come with a nice [paper](https://arxiv.org/abs/2005.04322) which does not require creating any account).

Once you have all the necessary files,
make sure that [Anaconda](https://www.anaconda.com/)
is installed, and run

```$ conda env create -f=environment.yml -n minimal_webapp```

to create a Conda+Pip environment named ```minimal_webapp``` with all the dependencies for the web app.
Next you need to activate the environment like

```$ conda activate minimal_webapp```

To run the web app locally and in development mode:

```$ FLASK_APP=app FLASK_ENV=development flask run```
 
After initialization succeeds, you can hit the app using your web browser at http://localhost:5000 (again, not https).

## Build and Push Docker Image

To containerize your web app,
first create a [Docker Hub](https://hub.docker.com/) account and note your username.
Next, run the following from the ```SpanglishModelingFlask/``` directory:

```$ docker build -t yourdockerhubusername/spanglish-modeling-app:latest .```

This takes approximately 6 minutes on my mid-2015 Macbook Pro (on *my* latest version).
Then you may push the built Docker image to Docker Hub using

```$ docker image push -a yourdockerhubusername/spanglish-modeling-app```

Once this is successful, you may follow the instructions in the previous section for deployment,
making sure to change `sarioz` to `yourdockerhubusername`.

## Train and Evaluate

To replicate the
[project results](https://docs.google.com/spreadsheets/d/1PwbSxT5r1alqZVMPIM7L0D00pduHuVTRr8YP8eLYMAs/edit?usp=sharing)
or tweak the training algorithms, step into
```Capstone-Project/``` then descend into any of the 5 subdirectories:
```POSTagger-RNN```, ```POSTagger-BERT```, ```SA-RNN```, ```SA-BERT```, or ```SA-Lexicon```.
The ones chosen for the web app are ```POSTagger-RNN``` and ```SA-RNN```.

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

A superset of the required dependencies for experimentation and model building can be found in 
```Capstone-Project/training_environment.yml```.
In order to automatically reconstruct this environment, run

```$ conda env create -f=training_environment.yml -n training-environment```

Within each of the 5 subprojects, ```main_training.py``` is the entry point for training and
```main_inference.py``` is the entry point for evaluating a specific model.

Lastly, if you want to experiment with ```SA-Lexicon```,
you need the files ```senticon.en.xml``` and ```senticon.es.xml```,
which you may download [here](http://www.lsi.us.es/~fermin/ML-SentiCon.zip),
under ```Capstone-Project/ML-Senticon/```.