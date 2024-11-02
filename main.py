import numpy as np
from typing import Optional
from enum import Enum


class State(Enum):
    SOLVED = 0
    UNSOLVED = 1
    INAPPLICABLE = 2


class Result:
    state: State
    objective_function_value: Optional[np.float64]
    solution: Optional[np.array]
    maximize: bool
    def __init__(self,
                 state: State,
                 objective_function_value: Optional[np.array] = None,
                 solution: np.float64 = None, maximize:bool = True):
        self.state = state
        self.objective_function_value = objective_function_value
        self.solution = solution
        self.maximize = maximize
#def print_initial_inputs(Vector &C, Matrix &A, Vector &b, double eps, bool maximize)
def print_initial_inputs(
        C: np.array,  # Vector of objective function coefficients
        A: np.array,  # Matrix of constraint coefficients
        x_0: np.array,  # Initial point (vector)
        b: np.array,  # Vector of right-hand side values of constraints
        eps: np.float64 = 0.01,  # Solution accuracy
        alpha: np.float64 = 0.5,  # Step coefficient
        maximize: bool = True):

    print("Running for the following inputs:\n")
    print(f"epsilon: {eps} ")
    print(f"alpha: {alpha}")
    print(f"x_0: {x_0} \n")
    if (maximize):
    
        print("Maximize")
    
    else:
    
        print("Minimize")
    
        
    z_str = "z = "
    previousIsZero = True
    lastNonZero = False
    for i in range(len(C)):
    
        isNegative = False;
        
        for k in range(len(C)):
        
            if (C[k] == 0):
            
                lastNonZero = True
            else:
                lastNonZero = False
                break
            
                

        if (not previousIsZero and not lastNonZero):
            z_str += " + ";
        

        if (C[i] != 0):
            if (C[i] != 1):
                if (C[i] < 0):
                    isNegative = True
                    z_str += "("
                
                z_str += str(C[i]) + " * "
            
            z_str += "x" + str(i + 1)
            if (isNegative):
                z_str += ")"
            previousIsZero = False
        else:
            previousIsZero = True
        
    print(z_str)
    print("\nsubject to the constrains:\n");
    for i in range(len(b)):
        c_str = ""
        previousIsZero = True
        lastNonZero = False
        for j in range(len(A[i])):
        #for (int j = 0; j < A.getColumns(); j++)
            isNegative = False
            for k in range(j, len(A[i])):
            #for (int k = j; k < A[i].size(); k++)
            
                if (A[i][k] == 0):
                
                    lastNonZero = True
                else:
                    lastNonZero = False
                    break

            if (not previousIsZero and not lastNonZero):
                c_str += " + "
            

            if (A[i][j] != 0): 
                if (A[i][j] != 1):
                    if (A[i][j] < 0):
                        isNegative = True
                        c_str += "("
                    
                    c_str += str(A[i][j]) + " * "
                
                c_str += "x" + str(j + 1)
                if (isNegative):
                    c_str += ")"
                
                previousIsZero = False
            else:
                previousIsZero = True
            
        
        c_str += " <= " + str(b[i])
        print(c_str)


def print_result(result:Result):

    if (result.state == State.INAPPLICABLE):
        print("The method is not applicable!")
    elif (result.state == State.UNSOLVED ):
        print("Unsolved problem!")
    else:
        print("SOLVED!")
        decVar_str = ""
        decVar_str +="Decision variables: ["
        for i in range(len(result.solution)):
        #for (int i = 0; i < result.solution.size(); i++)
            decVar_str += str(result.solution[i])
            if (i != len(result.solution) - 1):
            
                decVar_str += ", "
            
        decVar_str += "]"
        print(decVar_str)
        res_str = ""
        if (result.maximize):
        
            res_str += "Maximum "
        
        else:
        
            res_str += "Minimum "
        
        res_str += f"objective function value: {result.objective_function_value}"
        print(res_str)
    return 0


def interior_point(
        C: np.array,  # Vector of objective function coefficients
        A: np.array,  # Matrix of constraint coefficients
        x_0: np.array,  # Initial point (vector)
        b: np.array,  # Vector of right-hand side values of constraints
        eps: np.float64 = 0.01,  # Solution accuracy
        alpha: np.float64 = 0.5,  # Step coefficient
        maximizing: bool = True) -> Result:  # Flag for maximization or minimization
    # Check if the method is applicable: the initial point must satisfy the constraints
    if (not np.all(np.dot(A, x_0) >= b) or np.any(x_0 == 0)):
        return Result(State.INAPPLICABLE, maximize=maximizing)
    # If the problem is a minimization, invert the coefficients of the objective function
    if (not maximizing):
        C = -C
        
    m = len(A)  # Number of constraints
    n = len(A[0])  # Number of variables

    # Initialize variables for the initial iteration
    x = np.ones(n)  # Solution vector
    s = np.ones(m)  # Slack variables vector

    iteration = 0  # Iteration counter

    while(True):
        # Calculate slack variables for each constraint
        for i in range(m):
            slack = b[i]
            for j in range(n):
                slack -= A[i][j] * x[j]
            s[i] = slack
        
        # Update solution variables
        for i in range(min(m, n)):
            x[i] = s[i]

        # Create a diagonal matrix from the slack variables vector
        D = np.diag(s)

        # Solve the system of equations to find x*
        x_star = np.dot(np.linalg.inv(D), x)
        A_star = np.dot(A, D)
        C_star = np.dot(D, C)

        # Form the projection matrix
        I = np.eye(n)
        A_star_transpose = np.transpose(A_star)
        P = I - np.dot(A_star_transpose, np.linalg.inv(np.dot(A_star, A_star_transpose)))
        P = np.dot(P, A_star)

        # Calculate the gradient of the objective function
        C_p = np.dot(P, C_star)
        Mu = np.max(np.absolute(C_p))

        # Check the stopping criterion based on accuracy
        if Mu < eps:
            result = np.dot(C, x)
            return Result(State.SOLVED, objective_function_value=result, solution=x, maximize=maximizing)

        iteration += 1

        # Check the iteration limit
        if iteration >= 1000:
            return Result(State.UNSOLVED, maximize=maximizing)

        # Update the value of x* considering the step size and gradient
        x_star += (alpha / Mu) * C_p
        x = np.dot(D, x_star)


# TODO 5 tests (from assignment 1) and comparison with simplex and alpha = 0.9
def TEST_CASE_GENERAL():
    
    print("----------------------------RUNNING_TEST_GENERAL_CASE----------------------------")
    C = [5, 4]
    A = [
        [6, 4],
        [1, 2],
        [-1, 1],
        [0, 1]]
    b = [24, 6, 1, 2]
    x_0 = [1, 1]
    print_initial_inputs(C, A, x_0, b, 0.01, True);
    result = interior_point(C, A, x_0, b );
    
    
    if result.state == State.SOLVED:
        print_result(result)
    else:
        state_name = ""
        if result.state == State.UNSOLVED:
            state_name = "UNSOLVED"
        elif result.state == State.INAPPLICABLE:
            state_name = "INAPLICABLE"
        elif result.state == State.SOLVED:
            state_name = "SOLVED"
        print(f"incorrect state type. expected SOLVED, got {state_name}.")
        
        
    
    '''if (!(result.state == bounded))
    {
        std::string state_name;
        switch (result.state)
        {
        case unsolvable:
            state_name = "unsolvable";
            break;
        case unbounded:
            state_name = "unbounded";
            break;
        default:
            state_name = "bounded";
            break;
        }
        std::cout << "Incorrect state type. Expected bounded. Got " << state_name << std::endl;
        return 0;
    }

    if (!check_eq(result.objective_function_value, 21))
    {
        std::cout << "Incorrect objective function value. Expected 21. Got "
                  << result.objective_function_value << std::endl;
        return 0;
    }

    if (!(check_eq(result.solution[0], 3) && check_eq(result.solution[1], 1.5)))
    {
        std::cout << "Incorrect desire variables. Expected 3 and 1.5. Got "
                  << result.solution;
        return 0;
    }

    printResult(result);
    return 1;'''
    
    
    pass

TEST_CASE_GENERAL()