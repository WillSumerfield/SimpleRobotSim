from agent import train_agent
from perterbation_testing import perterbation_testing


def main():
    env = "Grasper/Manipulation-v0"
    for hand_type in [3]:
        for task_type in [0, 6]:
            print(f"Training agent for hand type {hand_type} and task type {task_type}")
            perterbation_testing(env, hand_type, task_type)

    perterbation_testing(env, 0, 0)

if __name__ == "__main__":
    main()