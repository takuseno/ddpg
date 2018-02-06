## DDPG
Deep Deterministic Policy Gradient implementation with Tensorflow.

## requirements
- Python3

## dependencies
- tensorflow
- gym[atari]
- opencv-python
- git+https://github.com/imai-laboratory/lightsaber

## usage
### training
```
$ python train.py --render
```

### playing
```
$ python train.py --render --load {path of models} --demo
```

### implementation
This is inspired by following projects.

- [DQN](https://github.com/imai-laboratory/dqn)
