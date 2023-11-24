# SCH_Applied_ai_assignment_2

DV2618/DV2619 – Assignment 2 - Genetic Algorithm, Data
Science and Implementations
This assignment has 4 different tasks and requires you to apply your knowledge of Genetic
Algorithms, Graph Search, Data science and implementation (preferably in Python). These topics
are covered in the learning materials in Lectures 4, 5, 6 and 7 and it is strongly recommended that
you work through the presentations from these weeks.
The deadline for completing this assignment is the 9th October 2023, 23:59 pm. You should
submit your work (e.g., zip file including your report, implementation etc.) on Canvas course
webpage. Note that, the zip file should be named with your name and surname (e.g.,
name_surname.zip).
Genetic Algorithms
Goal: To get acquainted with the idea and implementation of Genetic Algorithm (GA). To understand the
effects of parameters in the GA on an optimization problem. To do necessary changes and modifications
on the given implementation to resolve optimization problem.
Reading advice: Russell & Norvig: Artificial Intelligence - A Modern Approach, Section 4.3 in the 2nd
edition.

- task 3 is without Crossover
- task3v2 is with Crossover

# Tasks
## Tasks 1:
Problem Definition: You are going to solve a Travelling Salesman Problem (TSP). The TSP is defined as
finding the shortest route between a set of locations or cities that must be visited by a salesman. The
salesman‘s goal is to keep both the travel costs and the distance traveled as low as possible. Therefore, it
is necessary to design and implement a computer-based algorithm to find these set of locations that must
be visited by the salesman with a minimum cost. In this task, you will use a GA to resolve this optimization
problem and the implementation of the GA is provided on the Canvas.
You will test the performance of the GA using different combinations of population sizes and mutation
probabilities to understand and analyze the effects of parameters on the GA and you are going to change
the following parameter values to solve the corresponding problem:
a) Population size: 10, 20, 50, 100
b) Mutation Rate: 0.9, 0.6, 0.3, 0.1
c) Do not change the other parameters in the GA.
Note that, you will run your GA implementation one time for each parameter value to obtain the final
result at the end of the iteration of the algorithm.
* In the report, you must include, summarize, analyze, and discuss the results.
This is individual assessed assignment. You are allowed to discuss this assessment with
other students, but you should not copy their code, and you should not share your own
code with other students. Note that we will carry out plagiarism checks on all submissions.
## Tasks 2:
Problem Definition: You are going to solve a maximization problem and describe the given mathematical
equation. To do it, you must implement a fitness function the given equation below for the given GA
implementation (you can download the code from the Canvas). The GA will be used for optimizing the
equation which is as follows:

![image](https://github.com/MorrisSimons/SCH_Applied_ai_assignment_2/assets/38280463/145a6ecc-0928-471d-8027-de70dac73c0e)


* In the report, you must include only fitness function implementation. Also, you must submit your
whole python code with the report. Note that, your implementation must work properly and provide
correct output result.
## Tasks 3:
Problem Definition: You are going to solve a maze problem using Genetic Algorithm. The aim of this task
is to find a path in a maze, shown in the figure 1, for a mouse to reach the destination or food with a
shortest and acceptable path. You are going to implement a Genetic Algorithm that will find a solution
to find a path by optimizing the problem automatically.
In this task, the implementation should be done in Python and you should submit the implementation
in .py or ipynb.
Hint: To solve this problem, you need to create a 2D - matrix where the possible passes should be 1 and
obstacles must be a very large value (e.g., 1000). The matrix should be of the same size as the maze
matrix.
More information and details about the problem can be found on the following webpages:

https://www.educative.io/answers/what-is-the-maze-problem

https://tonytruong.net/solving-a-2d-maze-game-using-a-genetic-algorithm-and-a-search-part-2/

![image](https://github.com/MorrisSimons/SCH_Applied_ai_assignment_2/assets/38280463/3a38f4e3-83a2-475a-a301-58d8b15accf7)



Figure 1. Maze.
## Tasks 4:
Use Python software to complete this assignment. You can use sklearn, keras and/or tensorlow to
complete the assignment.
1. Regression for Data
1.1 Generate data points in Python using:
x_data = np.linspace(−0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise
a. Establish a linear regression model to predict y data (you can use sklearn).
b. Establish a polynomial model with highest power of 2 to predict y data (you can use sklearn).
c. Establish a three-layer (including input, 1 hidden (6 nodes) and output layer) neural network to
predict y data (keras or tensorflow is recommended). Split the dataset into training 80% and
testing data 20% using the following code in Python:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 0)
d. Calculate and compare mean squared errors of three models in 1.1 (a), (b) and (c).
e. Plot data points and curve of predictions in 1.1 (a) and 1.1 (b).
* In the report, briefly explain what you have done in (a), (b), (c), (d), and (e). Also, you must submit
your whole python code with the report.
Evaluation
The assessment criteria document for the Assignment 2 is provided on the Canvas course webpage under
the Assignment 2. You can contact the course responsible if you have an issue to access to the document
