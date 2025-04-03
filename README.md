# Stress Relief RL Agent

This project focuses on enhancing mental health awareness and accessibility for stress relief through a reinforcement learning (RL) approach. In the designed simulated environment the agent (representing a stressed individual) learns to navigate toward a meditation zone while avoiding obstacles symbolizing stress triggers like loud noise, overthinking, and phone notifications. The agent is trained  using a Deep Q-Network (DQN) agent and PPO, and demonstrates how AI can navigate therapeutic guidance to enhance mental health awareness and accessibility to stress relief techniques.

A Reinforcement Learning (RL) system for guiding users to therapeutic zones while avoiding stressors.

# Project Overview

This project trains Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents in a custom 2D environment where the goal is to navigate to stress-relief zones (meditation/safe areas) while avoiding obstacles.

# Technical Requirements

python 3.8+
gymnasium
stable-baselines3
pygame
numpy
tensorflow

# Installation

1. Ensure you have Python  installed.
2. Clone repository: `https://github.com/jeanraisa/Mental-AI-DQN.git`
3. Install dependencies: `pip install -r requirements.txt`

# Environment Details
State Space:

 * Obstacles 
 * Meditation zone 
 * Safe zone

Action Space: 

Four directional movements (up, down, left, right)

# Visual Elements

 * Blue Circle: Agent
 * Red Squares: Obstacles
 * Green Square: Meditation zone
 * Yellow : Safe zone 
