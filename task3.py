import random
from tqdm import tqdm
import copy
import sys
import matplotlib.pyplot as plt
import time

maze = [
    [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0]
]

# default functions
def get_starting_position(maze) -> tuple:
    """get starting position
    args = (maze = maze)
    returns = starting position (0,1) or (1,1) or (0,2)..."""
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 2:
                return (i, j)


def get_ending_position(maze) -> tuple:
    """get ending position
    args = (maze = maze)
    returns = ending position (0,1) or (1,1) or (0,2)..."""
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 3:
                maze[i][j] = 0
                return (i, j)


def get_possible_moves(maze, position) -> list:
    """get possible moves
    args = (maze = maze, position = current position (0,1))
    returns = list of possible moves ["up", "down", "left"...]
    """
    moves = []
    if position[0] > 0:
        if maze[position[0] - 1][position[1]] == 0 and (position[0] - 1, position[1]):
            moves.append("up")
    if position[0] < len(maze) - 1:
        if maze[position[0] + 1][position[1]] == 0 and (position[0] + 1, position[1]):
            moves.append("down")
    if position[1] > 0:
        if maze[position[0]][position[1] - 1] == 0 and (position[0], position[1] - 1):
            moves.append("left")
    if position[1] < len(maze[position[0]]) - 1:
        if maze[position[0]][position[1] + 1] == 0 and (position[0], position[1] + 1):
            moves.append("right")
    return moves

def get_new_position(position, move) -> tuple:
    """get new position
    args = (position = current  (0,1), move = "up", "down", "left", "right")
    returns = new position (0,1) or (1,1) or (0,2)..."""
    if move == "up":
        return (position[0] - 1, position[1])
    if move == "down":
        return (position[0] + 1, position[1])
    if move == "left":
        return (position[0], position[1] - 1)
    if move == "right":
        return (position[0], position[1] + 1)

def init_population(population_size, step_limit) -> list:
    """init population returns a list of genes, where each gene is a list of positions
    args = (population_size = number of population, step_limit = max number of steps)
    returns = [["up", "down", left...], gene2, ...]
    """
    population = [] # population = [gene1, gene2, ...]
    for _ in range(population_size):
        gene = [] # gene = ["up", "down", "left", "right", ...]
        for _ in range(step_limit):
            gene.append(random.choice(["up", "down", "left", "right"]))
        population.append(gene)
    return population


def fitness_eval(position, end_position, steps) -> int:
    """fitness evaluation function returns abs of x and y distance between position and end_position
    args = (position = current position, end_position = ending position, steps = number of steps)
    """
    return (closest_value(position, end_position) + steps)


def closest_value(position, end_position) -> int:
    """returns closest value to end position
    args = (position = current position(posx, posy), end_position = ending position(posx, posy))
    returns = closest value
    """
    return abs(position[0] - end_position[0]) + abs(position[1] - end_position[1])


def fitness_eval_population(population, end_position,start_position) -> list:
    """fitness evaluation population
    args = (population = list of genes, end_position = ending position)
    returns = list of tuples (gene index, fitness, number of steps, popultation)
    """
    fitness = []
    fittedgene = []
    closest_value_position = (0,0)
    for genes in range(len(population)):
        position = start_position
        best_pos = sys.maxsize
        step_counter = 0
        for move in population[genes]:
            move_possibliity = get_possible_moves(maze, position)
            step_counter += 1
            if move in move_possibliity:
                position = get_new_position(position, move)
                fittedgene.append(move)
                if position == end_position:
                    break # Break if we have reached the end
                close_val = closest_value(position, end_position)
                if best_pos > close_val:
                    best_pos = close_val
                    closest_value_position = position
        fitness.append((genes, fitness_eval(closest_value_position, end_position, step_counter), step_counter, population[genes]))
        step_counter = 0
        fittedgene = []
    return fitness


def selection(pop_with_fitness, population_loss, new_born_rate) -> list and int:
    """Gets the best genes from population
    popultaion size needs to be minimum 10
    args = (pop_with_fitness = list of tuples (gene index, fitness, number of steps, popultation), population_loss = number between 0 and 1)
    returns = list of best genes = [gene, gene, gene, ...]
    """
    best_genes = []
    # Sort to get best genes
    for _ in pop_with_fitness:
        sorted_pop = sorted(pop_with_fitness, key=lambda x: x[1])
    # population purge function amd append survivors to best_genes
    purge = int(round(len(sorted_pop)*(1 - population_loss)))
    new_born_rate = int(round(len(sorted_pop)*new_born_rate))
    # append survivors to best_genes
    for i in range(purge):
        best_genes.append(copy.deepcopy(sorted_pop[i][3]))
        #print("Survivor: ", sorted_pop[i][0], "Fitness: ", sorted_pop[i][1], "Steps: ", sorted_pop[i][2])
    # for plotting top genes later
    for i in range(len(best_genes)):
        #print("Gene: ", sorted_pop[i][0], "Fitness: ", sorted_pop[i][1], "Steps: ", sorted_pop[i][2])
        avg_top_fitness = sum([sorted_pop[i][1] for i in range(len(best_genes))]) / len(best_genes)
    # append newborns to best_genes newbors are of top genes
    for i in range(new_born_rate):
        #print("Newborn: ", sorted_pop[i][0], "Fitness: ", sorted_pop[i][1], "Steps: ", sorted_pop[i][2],"len in steps: ", len(sorted_pop[i][3]))
        best_genes.append(copy.deepcopy(sorted_pop[i][3]))
    return best_genes, avg_top_fitness


def cross_over_point_selection(smallest_gene) -> int and int:
    """Cross over point selection function"""
    position1 = random.randint(0, smallest_gene)
    position2 = random.randint(0, smallest_gene)
    return min(position1, position2), max(position1, position2)


def cross_over(gene1, gene2) -> list:
    """Cross over function the function takes two genes and returns 2 new genes
    args = (gene1 = (["right","left"...], 1), gene2 = (gene, 2))
    returns = 2 new genes
    """
    new_gene1 = [] # new gene
    new_gene2 = [] # new gene
    smallest_gene = min(len(gene1[0]), len(gene2[0]))
    start_point, end_point = cross_over_point_selection(smallest_gene)
    for i in range(smallest_gene):
        if i >= start_point and i <= end_point:
            new_gene1.append(gene2[0][i])
            new_gene2.append(gene1[0][i])
        else:
            new_gene1.append(gene1[0][i])
            new_gene2.append(gene2[0][i])
    return (new_gene1, gene1[1]), (new_gene2, gene2[1])


def mutation(population, mutaion_rate, random_mutation = False) -> list:
    """mutation on genes in population
    args = (population = list of genes, mutaion_rate = number between 0 and 1, if allow random mutation to create new gene sets)
    returns = mutated population [["up","left", "down"...], gene2, gene3, ...]
    """
    cross_over_lst = []
    
    for i in range(len(population)):
        random_value = random.random()
        if random_value < mutaion_rate and len(cross_over_lst) != 2:
            cross_over_lst.append((population[i], i))
        elif len(cross_over_lst) == 2:
            new_gene1, new_gene2 = cross_over(cross_over_lst[0], cross_over_lst[1])
            population[new_gene1[1]] = new_gene1[0]
            population[new_gene2[1]] = new_gene2[0] 
            cross_over_lst = []

        if random_mutation:
            for j in range(len(population[i])):
                random_value = random.random()
                if random_value < mutaion_rate:
                    population[i][j] = random.choice(["up", "down", "left", "right"])
                    
    return population


def start_plotting(chart_fitness_data, fig_filename = "fitness_data") -> None:
    """plots fitness over generations
    arg = (chart_fitness_data = list of tuples (generation[1,2,3,... ], avg_fitness[245, 542, 123,...]))
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    x, y = [], []
    for plots in chart_fitness_data:
        x.append(plots[0])
        y.append(plots[1])
    ax.plot(x,y, color="blue")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Fitness over generations")
    plt.show()
    fig.savefig(f"{fig_filename}.png")


def draw_maze(population,start_position, end_position,draw_int_mazes, mazefilename="maze.txt") -> None:
    """Draws an maze in txt file
    args = (population = list of genes, start_position, end_position, draw_int_mazes = number of mazes to draw)
    """
    maze_copy = copy.deepcopy(maze)
    for gene in range(draw_int_mazes):
        input(f"Drawing maze {gene}, steps: {population[gene][0]} press enter to continue...")
        position = start_position
        stepcounter = 0
        for move in population[gene][1]:
            moves = get_possible_moves(maze, position)
            if move in moves:
                stepcounter += 1
                position = get_new_position(position, move)
                maze_copy[position[0]][position[1]] = "-"
                if position == end_position:
                    break
        maze_copy[end_position[0]][end_position[1]] = "3"
        with open (mazefilename, "w") as f:
            for e in range(len(maze_copy)):
                for z in range(len(maze_copy[e])):
                    f.write(str(maze_copy[e][z]))
                f.write("\n")
        maze_copy = copy.deepcopy(maze)


def rank_gene(genes, start_pos, end_pos) -> list:
    """ranks the solution to prepare for drawing
    args = (best_genes = list of genes)
    returns = ([("gene", step_counter), ("gene", step_counter), ...]])
    """
    step_counter = 0
    genes_with_step = []
    for gene in genes:
        position = start_pos
        step_counter = 0
        for move in gene:
            moves = get_possible_moves(maze, position)
            if move in moves:
                step_counter += 1
                position = get_new_position(position, move)
                if position == end_pos:
                    break
        genes_with_step.append((step_counter,gene))
    for _ in genes_with_step:
        best_genes = sorted(genes_with_step, key=lambda x: x[0])
    return best_genes


if __name__ == "__main__":
    # Parameters
    draw_int_mazes = 5 # Number of mazes to draw
    generations = 50 # Number of generations
    popultaiton_size = 50 # Population size number of genes
    step_limit = 7750 # Max number of steps in a gene
    pop_loss_rate = 0.10 # Poploss are death of worst genes
    new_born_rate = 0.095 # Newborns are of top genes
    mutation_rate = 0.8 # Mutation rate is the chance of mutation
    learning_rate = 0.05 # Learning rate is the rate of mutation rate decrease
    thresholdrate = 0.0003 # Threshold is the minimum mutation rate
    ending_position = get_ending_position(maze)
    start_position = get_starting_position(maze)
    start = time.time()
    allow_random_mutations = True
    # Init population
    print("Settings:")
    print(f"Generations: {generations}      | popultaiton_size: {popultaiton_size}\nStep_limit: {step_limit}     | pop_loss_rate: {pop_loss_rate}\nNew_born_rate: {new_born_rate} | mutation_rate: {mutation_rate}\nLearning_rate: {learning_rate}  | thresholdrate: {thresholdrate}")
    print(f"Random mutations: {allow_random_mutations}")
    print("[INFO] Init population")
    population = init_population(popultaiton_size, step_limit)
    chart_fitness_data = []
    # Start
    print(f"Evolving {generations} generations...")
    for generation in tqdm(range(generations)):
        fitted_pop = fitness_eval_population(population, ending_position, start_position)
        # Selection
        best_genes, avg_fitness = selection(fitted_pop, pop_loss_rate, new_born_rate)
        chart_fitness_data.append((generation, avg_fitness))
        # Mutation
        population = mutation(best_genes, mutation_rate, random_mutation=allow_random_mutations)
        if mutation_rate > thresholdrate:
            mutation_rate = mutation_rate * (1- learning_rate)
    # End 
    end = time.time() - start
    if end > 60:
        print(f"Elapsed Time: {(end / 60):.2f} minutes")
    else:
        print(f"Elpased Time: {end:.2f} sec")
    # Plotting
    print("[INFO] Plottinging the average fitness over generations")
    start_plotting(chart_fitness_data)
    print(f"Drawing top {draw_int_mazes} mazes")
    print("[INFO] Drawing maze go to maze.txt to see drawing of maze")
    ranked_genes = rank_gene(best_genes, start_position, ending_position)
    draw_maze(ranked_genes,start_position, ending_position, draw_int_mazes)
    print(f"Best gene with this stepcounter: {ranked_genes[0][0]}:  ")
    save_mazesolution = input("[Y/N] Do you want to save the maze solution (['down', 'left'...]): ")
    if save_mazesolution == "Y" or save_mazesolution == "y":
        with open("mazesolution.txt", "w") as f:
            for i in range(len(ranked_genes[0][1])):
                f.write(str(ranked_genes[0][1][i]))
                f.write("\n")
    input("Press enter to exit...")
    print("[INFO] Done")
