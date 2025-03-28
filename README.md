# Gridworld Reinforcement Learning Simulator

## Overview
This is an interactive Python app that shows off different reinforcement learning algorithms on a 3x4 gridworld environment. It's just an exercise for my own personal education. The simulator provides a visual representation of how a few different algorithms learn and optimize policy and value functions using variations of the Bellman equation.

## Features
- Interactive GUI for visualizing reinforcement learning algorithms
- Supports multiple learning methods:
  - Value Iteration
  - Policy Iteration
  - Q-Learning
  - Epsilon-Greedy Learning

## Grid Environment
![image](https://github.com/user-attachments/assets/824586c8-cfcb-48ac-b7c7-0684b09d8c53)

- 3x4 grid with:
  - Terminal states (+1 and -1 rewards)
  - Wall state
  - Probabilistic state transitions, depending on value given in UI
  - Configurable parameters like step cost, transition probability, and discount factor

## User Interface
- Interactive controls for:
  - Algorithm selection
  - Parameter tuning (epsilon, alpha, discount, movement noisiness)
  - Visualization speed control
- Toggle between two display modes showing V-Scores and Optimal Policies, or Q-Scores of each action (move N, E, S, or W).

## Dependencies
- Python 3.11.3
- tkinter
