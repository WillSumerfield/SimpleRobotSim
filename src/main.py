"""
Users can play the Grasper environments, train a new agent, or test the current agent.
"""
import argparse

from manual_control import manual_control
from agent import Agent


def main():
    parser = argparse.ArgumentParser(description="Simple Robot Simulator")
    parser.add_argument("mode", choices=["test", "train", "manual"], help="Mode of operation")
    parser.add_argument("env", choices=["manipulation", "claw_game"], help="Mode of operation")
    parser.add_argument("--agent-save-file", type=str, help="File to save or load the agent", default="agent.pkl")
    args = parser.parse_args()

    if args.env == "manipulation":
        args.env = "Grasper/Manipulation-v0"

    if args.mode == "test":
        agent = Agent(args.env)
        agent.test()
        
    elif args.mode == "train":
        agent = Agent(args.env)
        agent.train()

    elif args.mode == "manual":
        manual_control(args.env)


if __name__ == "__main__":
    main()