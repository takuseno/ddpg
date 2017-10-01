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
$ python train.py --gpu {0 or -1} --render --final-steps 10000000
```

### playing
```
$ python play.py --gpu {0 or -1} --render --load {path of models}
```

### implementation
This is inspired by following projects.

- [DQN](https://github.com/imai-laboratory/dqn)
