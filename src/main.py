"""
Users can play the Grasper environments, train a new agent, or test the current agent.
"""
import argparse

from manual_control import manual_control
from agent import train_agent, test_agent


def main():
    parser = argparse.ArgumentParser(description="Simple Robot Simulator")
    parser.add_argument("mode", choices=["testing", "training", "manual"], help="Mode of operation")
    parser.add_argument("env", choices=["manipulation", "claw_game"], help="Mode of operation")
    parser.add_argument("--agent-save-file", type=str, help="File to save or load the agent", default="agent.pkl")
    args = parser.parse_args()

    if args.mode == "testing":
        train_agent()
        
    elif args.mode == "training":
        test_agent()

    elif args.mode == "manual":
        manual_control(args.env)


if __name__ == "__main__":
    main()