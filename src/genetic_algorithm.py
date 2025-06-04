import os
import random
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import gymnasium as gym
from deap import base, creator, tools
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from hand_morphologies import unnorm_hand_params
from agent import CPU_COUNT, PPO_ARGS, DAPG
import Grasper
from Grasper.wrappers import BetterExploration, TaskType


NUM_GENERATIONS = 20
MORPHOLOGY_SAVE_PATH = "morphologies/"
CHECKPOINTS_FOLDER = "checkpoints/genetic_algorithm"
LOGS_FOLDER = "training_logs/genetic_algorithm"
DIAGRAMS_FOLDER = "diagrams/genetic_algorithm"
MODEL_NAME = "policy.zip"
HAND_PARAM_NAME = "hand_params.npy"


def _make_env(env_id, task_type):
    env = gym.make(env_id)
    env = BetterExploration(env)
    if task_type is not None:
        env = TaskType(env, task_type)
    env = gym.wrappers.FlattenObservation(env)
    env = Monitor(env)
    check_env(env)
    return env


def streamgraph(data: np.ndarray, labels: list):
    # Compute cumulative stacks
    cumulative = np.cumsum(data, axis=0)
    total = cumulative[-1]  # Total height at each generation
    baseline = -0.5 * total  # Centering the stream vertically
    x = np.linspace(0, data.shape[1], data.shape[1])

    # Compute lower and upper bounds for each stream (band)
    lower = np.zeros_like(data)
    upper = np.zeros_like(data)

    for i in range(len(labels)):
        if i == 0:
            lower[i] = baseline
        else:
            lower[i] = upper[i-1]
        upper[i] = lower[i] + data[i]

    # Plot each stream layer
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(labels)):
        ax.fill_between(x, lower[i], upper[i], color=colors[i % len(colors)], label=f"Species {labels[i]}")
    
    plt.legend(loc="upper right")


class Individual():
    def __init__(self, index, hand_morphology: np.ndarray):
        self.index = index
        self.parent_index = None
        self.initialized = False
        self.hand_morphology = hand_morphology

    def __getitem__(self, index):
        return self.hand_morphology[index]

    def __setitem__(self, index, value):
        self.hand_morphology[index] = value

    def __len__(self):
        return len(self.hand_morphology)

    def __iter__(self):
        return iter(self.hand_morphology)

    def __repr__(self):
        return f"Invidual#{self.index}({self.hand_morphology})"


class GeneticAlgorithm():

    CROSSOVER_PROB = 0.7
    MUTATION_PROB = 0.5
    MUTATION_STD = 0.5
    POP_SIZE = 5
    TIMESTEPS_PER_GENERATION = 3e5
    REWARD_THRESHOLD = 1000
    EVALUATION_EPISODES = 100


    def __init__(self, env_id: str, task_type: int = None):
        
        # Setup the environment and policy network info
        self.task_type = task_type
        self.env = make_vec_env(lambda: _make_env(env_id, self.task_type), n_envs=CPU_COUNT, vec_env_cls=SubprocVecEnv)

        self.tensorboard_log = f"{LOGS_FOLDER}/multitask" if self.task_type is None else f"{LOGS_FOLDER}/task_{self.task_type}"
        self.task_count = self.env.get_attr("OBJECT_TYPES")[0]
        self.MODEL_ARGS = {**PPO_ARGS, "tensorboard_log": self.tensorboard_log}
        self.MODEL_ARGS["verbose"] = 0
        self.MODEL_ARGS["policy_kwargs"]["task_count"] = self.task_count
 
        # Define the fitness
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", Individual, fitness=creator.FitnessMin)

        self.individual_index = 0
        def init_individual():
            return creator.Individual(self.get_new_index(), np.random.normal(0, 1, size=self.task_count).astype(np.float32))

        # Define the population
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", init_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Define the common functions
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("crossover", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=self.MUTATION_STD, indpb=0.2)
        self.toolbox.register("select", tools.selNSGA2)  # NSGA-II selection

        # Create the initial population
        self.generation = 0
        self.evaluated_indices = set() # Only train individuals once per generation
        self._pop = self.toolbox.population(n=self.POP_SIZE)
        self.evaluate_population(self._pop)

        # Graph data
        self.data_dict = dict({"parent": None, "top fitness": -1, "generation": [], "proportion": [], "fitness": []})
        self.species_data = dict()
        os.makedirs(DIAGRAMS_FOLDER, exist_ok=True)
        for individual in self._pop:
            self.species_data[individual.index] = copy.deepcopy(self.data_dict)
            self.species_data[individual.index]["top fitnesss"] = individual.fitness.values[0]
            self.species_data[individual.index]["generation"] += [0]
            self.species_data[individual.index]["proportion"] += [1/self.POP_SIZE]
            self.species_data[individual.index]["fitness"] += [individual.fitness.values[0]]
        self._update_graphs()


    def get_new_index(self):
        index = self.individual_index
        self.individual_index += 1
        return index


    def run_generation(self) -> np.ndarray:
        self.generation += 1
        self.evaluated_indices = set()

        offspring = list(map(self.toolbox.clone, self._pop))
        random.shuffle(offspring)

        # Crossover
        # for child1, child2 in zip(offspring[::2], offspring[1::2]):
        #     if self.toolbox.random() < self.CROSSOVER_PROB:
        #         self.toolbox.crossover(child1, child2)
        #         del child1.fitness.values
        #         del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if np.random.rand() < self.MUTATION_PROB:
                mutant.parent_index = mutant.index
                mutant.index = self.get_new_index()
                mutant.initialized = False
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate offspring have had variation
        self.evaluate_population(offspring + self._pop)

        # Count the number of individuals per species
        species_counts = dict()
        for individual in (offspring+self._pop):
            if individual.index not in species_counts:
                species_counts[individual.index] = 1
            else:
                species_counts[individual.index] += 1

        # Replace the old population with the new one
        self._pop[:] = self.toolbox.select(offspring + self._pop, self.POP_SIZE)

        # Update the species data
        for species in species_counts:
            if species not in self.species_data:
                self.species_data[species] = copy.deepcopy(self.data_dict)
            self.species_data[individual.index]["generation"] += [self.generation]
            self.species_data[individual.index]["proportion"] += [species_counts[species]/self.POP_SIZE]
            self.species_data[individual.index]["fitness"] += [individual.fitness.values[0]]
            self.species_data[individual.index]["parent"] = individual.parent_index
            self.species_data[individual.index]["max fitness"] = max(self.species_data[individual.index]["fitness"])

        self._update_graphs()


    def evaluate_population(self, population):
        for idx, individual in enumerate(population):
            # We need to evaluate the entire population and each mutated offspring bc. they all need to keep trianing the same amount to keep it fair.
            if individual.index not in self.evaluated_indices:
                # Reset the environment with the new hand morphology
                options = {"hand_parameters": unnorm_hand_params(individual.hand_morphology.copy())}
                self.env.set_options(options)
                self.env.reset()
                print(f"Evaluation Progress: {int(100*idx/len(population))}%      ", end='\r')
                individual.fitness.values = self.toolbox.evaluate(individual)
                self.evaluated_indices.add(individual.index)


    def get_fitness_metrics(self):
        avg_fitness = np.array([individual.fitness.values[0] for individual in self._pop]).mean()
        max_fitness = np.array([individual.fitness.values[0] for individual in self._pop]).max()
        min_fitness = np.array([individual.fitness.values[0] for individual in self._pop]).min()
        return avg_fitness, max_fitness, min_fitness


    def _evaluate(self, individual: Individual):
        # Load last checkpoint
        if individual.initialized:
            args = self.MODEL_ARGS.copy()
            args["path"] = f"{CHECKPOINTS_FOLDER}/{individual.index}/{MODEL_NAME}"
            model = DAPG.load(env=self.env, **args)
        else:
            # Load the parent's model
            if individual.parent_index is not None:
                args = self.MODEL_ARGS.copy()
                args["path"] = f"{CHECKPOINTS_FOLDER}/{individual.parent_index}/{MODEL_NAME}"
                model = DAPG.load(env=self.env, **args)
            # Create a new model
            else:
                args = self.MODEL_ARGS.copy()
                model = DAPG(env=self.env, **args)
            individual.initialized = True

        # Train model
        model.learn(total_timesteps=self.TIMESTEPS_PER_GENERATION)

        # Save model and hand parameters
        if not os.path.exists(f"{CHECKPOINTS_FOLDER}/{individual.index}"):
            os.makedirs(f"{CHECKPOINTS_FOLDER}/{individual.index}")
        model.save(f"{CHECKPOINTS_FOLDER}/{individual.index}/{MODEL_NAME}")
        np.save(f"{CHECKPOINTS_FOLDER}/{individual.index}/{HAND_PARAM_NAME}", individual.hand_morphology)
        
        # Evaluate model
        mean_reward, std_reward = evaluate_policy(model, self.env, n_eval_episodes=self.EVALUATION_EPISODES, deterministic=True)

        return mean_reward,


    def _update_graphs(self):

        # Streamgraph
        y_data = np.zeros((len(self.species_data), self.generation+1))
        for index, species in self.species_data.items():
            species_prop = species["proportion"]
            species_generations = species["generation"]
            y_data[index, species_generations] = species_prop
        labels = list(self.species_data.keys())
        streamgraph(y_data, labels)
        plt.title("Species Proportions")
        plt.xlabel("Generation")
        plt.ylabel("Proportion")
        plt.savefig(f"{DIAGRAMS_FOLDER}/species_proportions.png")
        plt.clf()

        # Heritage Graph
        G = nx.DiGraph()
        G.add_nodes_from([idx for idx in self.species_data.keys()])
        G.add_edges_from([(idx, species["parent"]) for idx, species in self.species_data.items() if species["parent"] is not None])
        pos = {idx: (idx, min(species["generation"])) for idx, species in self.species_data.items()}
        nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
        plt.title("Heritage Graph")
        plt.axis('off')
        plt.savefig(f"{DIAGRAMS_FOLDER}/heritage_graph.png")
        plt.clf()

        # Fitness per species
        generations = []
        fitness = []
        species_indices = []
        for idx, species in self.species_data.items():
            generations += species["generation"]
            fitness += species["fitness"]
            species_indices += [idx]*len(species["generation"])
        data = pd.DataFrame({
            "Generation": generations, 
            "Fitness": fitness, 
            "Species Index": species_indices
        })
        sns.lineplot(data=data, x="Generation", y="Fitness", hue="Species Index")
        plt.title("Fitness per Species")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.savefig(f"{DIAGRAMS_FOLDER}/fitness_per_species.png")
        plt.clf()


def evolve_hands(env_id: str, task_type):

    print(f"Starting Co-Evolution with {GeneticAlgorithm.POP_SIZE} individuals")
    genetic_algorithm = GeneticAlgorithm(env_id, task_type)
    avg, mx, mn = genetic_algorithm.get_fitness_metrics()
    print(f"Average Starting Fitness: {avg:.2f}")

    for generation_idx in range(NUM_GENERATIONS):
        genetic_algorithm.run_generation()
        avg, mx, mn = genetic_algorithm.get_fitness_metrics()
        print(f"Generation {generation_idx} Avg. Fitness: {avg:.2f}, Max: {mx:.2f}, Min: {mn:.2f}          ")

    print(f"Co-Evolution Finished")
    avg, mx, mn = genetic_algorithm.get_fitness_metrics()
    print(f"Avg. Fitness: {avg:.2f}, Max: {mx:.2f}, Min: {mn:.2f}                                          ")

    # Save the top hands
    save_path = MORPHOLOGY_SAVE_PATH + f"_task_{task_type}.npy"
    final_morphologies = np.array([individual.hand_morphology for individual in genetic_algorithm._pop])
    os.makedirs(MORPHOLOGY_SAVE_PATH, exist_ok=True)
    np.save(save_path, final_morphologies)
    print(f"Saving top hands to {save_path}")
