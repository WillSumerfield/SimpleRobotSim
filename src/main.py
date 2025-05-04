"""
Users can play the Grasper environments, train a new agent, or test the current agent.
"""
import argparse

from manual_control import manual_control
from agent import train_agent, test_agent, param_sweep, convert_to_baseline, get_video
from demo import provide_demos, play_demos
from baseline import train_baseline, test_baseline
from genetic_algorithm import evolve_hands


def main():
    parser = argparse.ArgumentParser(description="Simple Robot Simulator")
    parser.add_argument("mode", choices=["genetic-algorithm",
                                         "test-agent", "train-agent", "param-sweep-agent",
                                         "manual", "demo", "get-video",
                                         "train-baseline", "test-baseline", "convert-to-baseline"], 
                                         help="Mode of operation")
    parser.add_argument("env", choices=["manipulation", "claw_game"], help="Mode of operation")
    parser.add_argument("--continue-training", action="store_true", help="Use the latest checkpoint to continue training")
    parser.add_argument("--agent-save-file", type=str, help="File to save or load the agent", default="agent.pkl")
    parser.add_argument("--demo-index", type=int, help="The demo index to train the baseline on.", default=1)
    parser.add_argument("--replay", type=int, help="Playback the provided demo collection number.")
    parser.add_argument("--hand-type", type=int, help="The index of the pre-defined hand to use. Claw default.", default=0)
    parser.add_argument("--task-type", type=int, help="The index of the subtask to train/test on. Defaults to all objects.", default=None)
    parser.add_argument("--use-model", action="store_true", help="Use the model as opposed to a checkpoint to test.")
    parser.add_argument("--ga", type=int, help="The individual index # of the Genetic Algorithm to view/test", default=None)
    args = parser.parse_args()

    if args.env == "manipulation":
        args.env = "Grasper/Manipulation-v0"

    if args.mode == "genetic-algorithm":
        evolve_hands(args.env, args.task_type)

    if args.mode == "test-agent":
        test_agent(args.env, args.hand_type, args.task_type, checkpoint=(not args.use_model))
        
    elif args.mode == "train-agent":
        train_agent(args.env, args.hand_type, args.task_type, continue_training=args.continue_training)

    elif args.mode == "param-sweep-agent":
        param_sweep(args.env, args.hand_type, args.task_type)

    elif args.mode == "manual":
        manual_control(args.env, args.hand_type, args.task_type, args.ga)
    
    elif args.mode == "demo":
        if args.replay:
            play_demos(args.env, args.hand_type, args.replay)
        else:
            provide_demos(args.env, args.hand_type, args.task_type)

    elif args.mode == "get-video":
        get_video(args.env, args.hand_type, args.task_type, args.ga)
    
    elif args.mode == "train-baseline":
        train_baseline(args.env, args.demo_index)

    elif args.mode == "test-baseline":
        test_baseline(args.env, args.hand_type, args.task_type)
    
    elif args.mode == "convert-to-baseline":
        convert_to_baseline(args.env, args.hand_type, args.task_type, checkpoint=(not args.use_model))


if __name__ == "__main__":
    main()