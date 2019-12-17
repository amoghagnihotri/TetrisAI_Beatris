# TetrisAI vs Beatris

TetrisAI is a bot that plays [tetris](https://en.wikipedia.org/wiki/Tetris) using deep reinforcement learning, and Beatris is an adversarial network that learns to feed TetrisAI the worst pieces. Read more about it in our [Medium article](https://medium.com/@amoghhgoma/88fee6b068?).

## Results

TetrisAI consistently achieves a high score of 135 against random pieces. We have seen scores as high as over 300,000.

Beatris is able to stunt TetrisAI's score down to an average of ~19, with a total max score of 45.

#### Requirements

- Tensorflow (`tensorflow-gpu==1.14.0`, CPU version can be used too)
- Tensorboard (`tensorboard==1.14.0`)
- Keras (`Keras==2.2.4`)
- Opencv-python (`opencv-python==4.1.0.25`)
- Numpy (`numpy==1.16.4`)
- Pillow (`Pillow==5.4.1`)
- Tqdm (`tqdm==4.31.1`)

## To Run:

### Running TetrisAI vs random pieces

Make sure to have the flags set as such:
  
    trainingAgent = False
    trainingHater = False
    
And comment out the following lines of code:

    # hateris = DQNAgent(env.get_state_size(),
    #                  n_neurons=n_neurons, activations=activations,
    #                  epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
    #                  discount=discount, replay_start_size=replay_start_size,
    #                  training=trainingHater, agent_save_filepath=hater_save_filepath)
    # env.hater = hateris

### Running TetrisAI vs Beatris

Make sure to have the flags set as such:
  
    trainingAgent = False
    trainingHater = False
    
And have the following lines of code in:

    hateris = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size,
                     training=trainingHater, agent_save_filepath=hater_save_filepath)
    env.hater = hateris
    
### Training your own TetrisAI or Beatris

Set either `<trainingAgent>` or `<trainingHater>` flags to True depending on which one you are training.

Set a new file save name for `<agent_save_filepath>` or `<hater_save_filepath>` to mark the name of your resulting h5 file containing the trained agents.

Set desired `<epsilon_stop_episode>` and `<episodes>` depending on training time and explore vs eploit tradeoff.

### Team

This project was made as a project for Dr. Caramanis' EE460J Data Science Laboratory class at The University of Texas at Austin. The team consisted of [Amogh Agnihotri](amoghagnihotri@utexas.edu), Ramya Rajasekaran, Reese Costis, Prajakta Joshi, Rebecca Phung, and Suhas Raja.
