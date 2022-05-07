# art-CAN
Generating artwork using creative adversarial networks

Copy and pasted from HW5, requirements.txt probably has stuff we dont need

Notes to Self + For GCP:
-If Code isn't running locally or on GCP for some reason, might be because the GPU check in main.py
-The intermediate results folder name is based off hyperparameters
-Don't use virtual environment, the GCP should have the packages needed
-Folder Structure should look similar to this (data is outside the directory)
  data/
    wikiart_ultra_slim/
        label1/ 
        xxx.png
        xxy.png
        xxz.png
  art-Can/
    code/
        main.py
        ...

Resources:
- https://arxiv.org/abs/1706.07068
- https://www.tensorflow.org/tutorials/generative/dcgan
- https://github.com/otepencelik/GAN-Artwork-Generation

GAN Training Tips: 
- https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628
- https://github.com/soumith/ganhacks