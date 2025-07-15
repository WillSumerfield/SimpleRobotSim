"""
Users can play the Grasper environments, train a new agent, or test the current agent.
"""
import argparse

from src import ENV_2D, ENV_2_5D
from src.manual_control import manual_control
from src.agent import train_agent, test_agent, param_sweep, convert_to_baseline, get_video, get_photo
from src.genetic_algorithm import evolve_hands
from src.perterbation_testing import perterbation_testing


def main():
    parser = argparse.ArgumentParser(description="Simple Robot Simulator")
    parser.add_argument("mode", choices=["perterbation-test", "genetic-algorithm",
                                         "test-agent", "train-agent", "param-sweep-agent",
                                         "manual", "get-video", "get-photo",
                                         "convert-to-baseline", "test-baseline"], 
                                         help="Mode of operation")
    parser.add_argument("dimensionality", choices=["2D", "2.5D"], help="Dimensionality of the environment. There are differences between the inputs/outputs of the environments.")
    parser.add_argument("--hand-type", type=int, help="The index of the pre-defined hand to use. Claw default.", default=0)
    parser.add_argument("--task-type", type=int, help="The index of the subtask to train/test on. Defaults to all objects.", default=0)
    parser.add_argument("--use-model", action="store_true", help="Use the model as opposed to a checkpoint to test.")
    parser.add_argument("--ga", type=int, help="The individual index # of the Genetic Algorithm to view/test", default=None)
    parser.add_argument("--pt", type=int, help="The permutation iteration # to view", default=None)
    args = parser.parse_args()

    # Environment Choice
    env = None
    if args.dimensionality == "2D":
        env = ENV_2D
    elif args.dimensionality == "2.5D":
        env = ENV_2_5D
    else:
        raise ValueError("Invalid dimensionality. Choose '2D' or '2.5D'.")
    
    # Run Options
    if args.mode == "genetic-algorithm":
        evolve_hands(env, args.task_type)

    elif args.mode == "perterbation-test":
        perterbation_testing(env, args.hand_type, args.task_type)

    elif args.mode == "test-agent":
        test_agent(env, args.hand_type, args.task_type, checkpoint=(not args.use_model))
        
    elif args.mode == "train-agent":
        train_agent(env, args.hand_type, args.task_type)

    elif args.mode == "param-sweep-agent":
        param_sweep(env, args.hand_type, args.task_type)

    elif args.mode == "manual":
        manual_control(env, args.hand_type, args.task_type, args.ga)

    elif args.mode == "get-video":
        get_video(env, args.hand_type, args.task_type, args.ga, args.pt)

    elif args.mode == "get-photo":
        get_photo(env, args.hand_type, args.task_type, args.ga, args.pt)

    elif args.mode == "convert-to-baseline":
        convert_to_baseline(env, args.hand_type, args.task_type, checkpoint=(not args.use_model))

    else:
        raise ValueError("Invalid mode.")


if __name__ == "__main__":
    main()