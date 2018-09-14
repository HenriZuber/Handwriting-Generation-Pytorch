# Handwriting Generation

## Intro
This project is a direct application of the paper from Alex Graves on sequence generation.
- There are two parts to this project :
  -  Some fake handwriting generation
  -  A "converter" from string to handwritten

## How to use and how it was done

#### How it was done:
I mostly followed the paper without thinking about it too much.
I would really advise readers to take a long look at the paper in order to better understand the code as it is not as commented as it should be.
Some of the code could be changed (like the number of layers of lstm in the conditional generation model which I fixed at two) but I didn't really have the time to do it.

#### Using it :
- Step 1: Docker-compose run boulot bash
- Step 2: go to the src folder
- Step 3: python main.py


Change the config file according to what you want to run, whether it's conditional or not and inference or training.
If you want to get specific text or choose the random seed you want to use for the inferences, check out the dummy.py  file.


#### Principal Resource
https://arxiv.org/pdf/1308.0850.pdf

