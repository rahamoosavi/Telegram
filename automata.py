"""Implementations of Wolfram's Rule 30 and Conway's
Game of Life on various meshes"""

import numpy as np
import scipy
import matplotlib.pyplot as plt


def rule_thirty(initial_state, nsteps, periodic=False):
    """
    Perform iterations of Wolfram's Rule 30 with specified boundary
    conditions.

    Parameters
    ----------
    initial_state : array_like or list
        Initial state of lattice in an array of booleans.
    nsteps : int
        Number of steps of Rule 30 to perform.
    periodic : bool, optional
        If true, then the lattice is assumed periodic.

    Returns
    -------

    numpy.ndarray
         Final state of lattice in array of booleans

    >>> rule_thirty([False, True, False], 1)
    array([True, True, True])

    >>> rule_thirty([False, False, True, False, False], 3)
    array([True, False, True, True, True])
    """

    # write your code here to replace return statement
    if isinstance(initial_state, list):
        initial_state = np.asarray(initial_state)
    # Define the game rule
    thirty_rule = {
            (True, True, True): False, (True, True, False): False,
            (True, False, True): False, (True, False, False): True,
            (False, True, True): True, (False, True, False): True, 
            (False, False, True): True, (False, False, False): False
            }
    # Initalize the state
    L = len(initial_state)
    current_state = np.full((nsteps, L), False)
    current_state[0, :] = initial_state
    current_step = 0
    while current_step <= nsteps - 2:  
        for i in range(1, L-1):
            current_state[current_step+1, i] = thirty_rule.get((
                    current_state[current_step, i-1],
                    current_state[current_step, i], 
                    current_state[current_step, i+1]
                    ))
        if periodic == True:
            current_state[current_step+1, 0] = thirty_rule.get((
                    current_state[current_step, L-1],
                    current_state[current_step, 0], 
                    current_state[current_step, 1]
                    ))
            current_state[current_step+1, L-1] = thirty_rule.get((
                    current_state[current_step, L-2], 
                    current_state[current_step, L-1], 
                    current_state[current_step, 0]
                    ))
        else:
            current_state[current_step+1, 0] = thirty_rule.get((False, current_state[current_step, 0], current_state[current_step, 1]))
            current_state[current_step+1, L-1] = thirty_rule.get((current_state[current_step, L-2], current_state[current_step, L-1], False))
        current_step += 1 
        
    return current_state[nsteps-1, :]


def life(initial_state, nsteps):
    """
    Perform iterations of Conway’s Game of Life.
    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array of booleans.
    nsteps : int
        Number of steps of Life to perform.
    Returns
    -------
    numpy.ndarray
         Final state of grid in array of booleans
    """

    # write your code here to replace return statement
    if isinstance(initial_state, list):
        initial_state = np.asarray(initial_state)

    rows = initial_state.shape[0]
    cols = initial_state.shape[1]
    current_state = np.full((nsteps, rows+2, cols+2), False)
    current_state[0, 1:rows+1, 1:cols+1] = initial_state[:, :]
    current_step = 0
    while current_step <= nsteps - 2:    
        for i in range(1, rows+1):
            for j in range(1, cols+1):
                neighbors_list = [
                        current_state[current_step, i-1, j-1],
                        current_state[current_step, i-1, j],
                        current_state[current_step, i-1, j+1],
                        current_state[current_step, i, j-1], 
                        current_state[current_step, i, j+1],
                        current_state[current_step, i+1, j-1],
                        current_state[current_step, i+1, j],
                        current_state[current_step, i+1, j+1]
                        ]
                num_alive = neighbors_list.count(True)
                if current_state[current_step, i, j] is True:
                    if num_alive == 2 or num_alive == 3:
                        current_state[current_step+1, i, j] = True
                    else:   
                        current_state[current_step+1, i, j] = False
                elif current_state[current_step, i, j] is False:
                    if num_alive == 3:
                        current_state[current_step+1, i, j] = True
                    else:   
                        current_state[current_step+1, i, j] = False
        current_step += 1
        return(current_state[nsteps-1, 1:rows+1, 1:cols+1])




def life_periodic(initial_state, nsteps):
    """
    Perform iterations of Conway's Game of Life on a doubly periodic mesh.

    Parameters
    ----------
    initial_state : array_like or list of lists
        Initial 2d state of grid in an array of booleans.
    nsteps : int
        Number of steps of Life to perform.

    Returns
    -------

    numpy.ndarray
         Final state of grid in array of booleans
    """

    # write your code here to replace return statement
    if isinstance(initial_state, list):
        initial_state = np.asarray(initial_state)

    rows = initial_state.shape[0]
    cols = initial_state.shape[1]
    current_state = np.full((nsteps, rows+2, cols+2), False)
    current_state[0, 1:rows+1, 1:cols+1] = initial_state
    current_state[0, 0, 1] = current_state[0, rows, 1]
    current_state[0, rows+1, :] = current_state[0, 1, :]
    current_state[0, :, 0] = current_state[0, :, cols]
    current_state[0, :, cols+1] = current_state[0, :, 1]
    current_step = 0
    while current_step <= nsteps - 2:    
        for i in range(1, rows+1):
            for j in range(1, cols+1):
                neighbors_list = [
                        current_state[current_step, i-1, j-1],
                        current_state[current_step, i-1, j],
                        current_state[current_step, i-1, j+1],
                        current_state[current_step, i, j-1],
                        current_state[current_step, i, j+1],
                        current_state[current_step, i+1, j-1],
                        current_state[current_step, i+1, j],
                        current_state[current_step, i+1, j+1]
                       ]
            num_alive = neighbors_list.count(True)
            if current_state[current_step, i, j] is True:
                if num_alive == 2 or num_alive == 3:
                    current_state[current_step+1, i, j] = True
                else:   
                    current_state[current_step+1, i, j] = False
            elif current_state[current_step, i, j] is False:
                if num_alive == 3:
                    current_state[current_step+1, i, j] = True
                else:   
                    current_state[current_step+1, i, j] = False     
        current_state[current_step+1 , 0, :] = current_state[current_step+1, rows, :]
        current_state[current_step+1, rows+1, :] = current_state[current_step+1, 1, :]
        current_state[current_step+1, :, 0] = current_state[current_step+1, :, cols]
        current_state[current_step+1, :, cols+1] = current_state[current_step+1, :, 1]
        current_step += 1
    return current_state[nsteps-1, 1:rows+1, 1:cols+1]


def lifehex(initial_state, nsteps):
    """
    Perform iterations of Conway's Game of Life on
    a hexagonal tessellation.

    Parameters
    ----------
    initial_state : list of lists
        Initial state of grid on hexagons.
    nsteps : int
        Number of steps of Life to perform.

    Returns
    -------

    list of lists
         Final state of tessellation.
    """

    # write your code here to replace return statement
    
    return NotImplemented


def life_generic(matrix, initial_state, nsteps, environment, fertility):
    """
    Perform iterations of Conway’s Game of Life for an arbitrary
    collection of cells.
    Parameters
    ----------
    matrix : 2d array of ints (o or 1)
        a boolean matrix with rows indicating neighbours for each cell
    initial_state : 1d array of bools
        Initial state vectr.
    nsteps : int
        Number of steps of Life to perform.
    environment : set of ints
        neighbour counts for which live cells survive.
    fertility : set of ints
        neighbour counts for which dead cells turn on.
    Returns
    -------
    array
         Final state.
    """

    # write your code here to replace return statement
    return NotImplemented


# Remaining routines are for plotting

def plot_array(data, show_axis=False,
               cmap=plt.get_cmap('bone'), **kwargs):
    """Plot a 1D/2D array in an appropriate format.

    Mostly just a naive wrapper around pcolormesh.

    Parameters
    ----------

    data : array_like
        array to plot
    show_axis: bool, optional
        show axis numbers if true
    cmap : pyplot.colormap or str
        colormap

    Other Parameters
    ----------------

    **kwargs
        Additional arguments passed straight to pyplot.pcolormesh
    """
    plt.pcolormesh(data, edgecolor='b', cmap=cmap, **kwargs)

    plt.axis('equal')
    if show_axis:
        plt.axis('on')
    else:
        plt.axis('off')


def plot_hexes(data, **kwargs):
    """Plot a hexagonal array.

    Parameters
    ----------
    data : list of lists
        The state of the Game of Life system.

    This uses pyplot.fill to plot closed hexagons.
    """

    colors = ['w', 'r']
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            plot_one_hex((i, j), edgecolor='k',
                         facecolor=colors[1*val], **kwargs)


def plot_one_hex(coord, **kwargs):
    """Plot a hexagon, optionally filled.

    Parameters
    ----------
    coord : array_like
       The 2d coordinates of the centre of the hexagon

    This uses pyplot.fill to plot closed hexagons.
    """
    x_c = np.array([1, 0, -1, -1, 0, 1])
    y_c = np.array([0.5, 1.0, 0.5, -0.5, -1.0, -0.5])

    plt.fill(np.sqrt(3.0)*(coord[1]-0.5*(coord[0] % 2))+x_c,
             -1.5*coord[0]+y_c, **kwargs)
