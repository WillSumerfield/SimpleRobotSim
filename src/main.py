"""
Users can play the Grasper environments, train a new agent, or test the current agent.
"""
import argparse

from manual_control import manual_control
from agent import train_agent, test_agent, param_sweep
from demo import provide_demos, play_demos
from baseline import train_baseline, test_baseline


def main():
    parser = argparse.ArgumentParser(description="Simple Robot Simulator")
    parser.add_argument("mode", choices=["test", "train", "manual", "demo", "train_baseline", "test_baseline", "param_sweep"], help="Mode of operation")
    parser.add_argument("env", choices=["manipulation", "claw_game"], help="Mode of operation")
    parser.add_argument("--continue-training", action="store_true", help="Use the latest checkpoint to continue training")
    parser.add_argument("--agent-save-file", type=str, help="File to save or load the agent", default="agent.pkl")
    parser.add_argument("--demo-index", type=int, help="The demo index to train the baseline on.", default=1)
    parser.add_argument("--replay", type=int, help="Playback the provided demo collection number.")
    parser.add_argument("--hand-type", type=int, help="The index of the pre-defined hand to use. Claw default.", default=0)
    parser.add_argument("--task-type", type=int, help="The index of the subtask to train/test on. Defaults to all objects.", default=None)
    args = parser.parse_args()

    if args.env == "manipulation":
        args.env = "Grasper/Manipulation-v0"

    if args.mode == "test":
        test_agent(args.env, args.hand_type, args.task_type)
        
    elif args.mode == "train":
        train_agent(args.env, args.hand_type, args.task_type, continue_training=args.continue_training)

    elif args.mode == "manual":
        manual_control(args.env, args.hand_type, args.task_type)
    
    elif args.mode == "demo":
        if args.replay:
            play_demos(args.env, args.hand_type, args.replay)
        else:
            provide_demos(args.env, args.hand_type, args.task_type)
    
    elif args.mode == "train_baseline":
        train_baseline(args.env, args.demo_index)

    elif args.mode == "test_baseline":
        test_baseline(args.env, args.hand_type, args.task_type)

    elif args.mode == "param_sweep":
        param_sweep(args.env, args.hand_type, args.task_type)

if __name__ == "__main__":
    main()