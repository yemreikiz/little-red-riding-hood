from __future__ import annotations

import random
from typing import Optional


from ..pathfinder.types import Visualiser
# from ..models.frontier import PriorityQueueFrontier
from ..pathfinder.models.node import Node, Node
from ..pathfinder.models.grid import Grid
from ..pathfinder.models.solution import NoSolution, Solution
from ..maze import Maze



class Wanderer:
    """
        a class that takes the solution path and starts visiting path nodes
        when it steps on the gray node, it receives 5 coordinates to pick flowers
        The coordinates are stored in a list and the list is visited one by one, the path from the girls position to the
        flower is calculated using the AStartSearch class
    """
    @staticmethod
    def walk(maze: Maze,
             solution: Solution,
             ):

        # Create Node for the source cell
        node = maze.grid.get_node(pos=maze.start)

        # Enter wolf the grey, always 5th step
        wolf_node = maze.grid.get_node(pos=solution.path[4])
        wolf_node.value = "W"

        flower_locations = []
        i = 1
        while True:
            node = maze.grid.get_node(pos=solution.path[i])
            i += 1
            # If reached destination point
            if node.value == "W":
                # Pick 5 flower location from grid that are not wall, start, goal, or wolf
                while len(flower_locations) < 5:
                    rand_x = random.randint(0, len(maze.maze) - 1)
                    rand_y = random.randint(0, len(maze.maze[0]) - 1)
                    f_node = maze.maze[rand_x][rand_y]
                    print(f_node.value)
                    if f_node.value == '1':
                        f_node.value = "F"
                        flower_locations.append(f_node)
                        # TODO: append the flower image here

                break
            else:
                # mark the node as girl
                node.value = "G"

            # end of while loop

        # test until here
        return flower_locations

        # start visiting the flowers using AStarSearch
        # current_location = wolf_node
        # next_location = None
        # for flower in flower_locations:
        #     next_location = flower
        #     grid.start = current_location
        #     grid.end = next_location
        #     maze.dr

