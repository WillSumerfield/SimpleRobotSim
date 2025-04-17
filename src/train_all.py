import argparse

from agent import train_agent


def main():
    env = "Grasper/Manipulation-v0"
    train_agent(env, 3, None, False)
    train_agent(env, 4, None, False)
    train_agent(env, 5, None, False)


if __name__ == "__main__":
    main()