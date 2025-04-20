from agent import train_agent


def main():
    env = "Grasper/Manipulation-v0"
    for hand_type in range(6):
        for task_type in range(5):
            print(f"Training agent for hand type {hand_type} and task type {task_type}")
            train_agent(env, hand_type, task_type, continue_training=False)


if __name__ == "__main__":
    main()