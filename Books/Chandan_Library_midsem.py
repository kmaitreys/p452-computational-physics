########################################################################################################################
import copy
import math

import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################

"""
SECTIONS OF THE LIBRARY

- BASIC FUNCTIONS
- MATRICES AND LINEAR ALGEBRA
- ROOT FINDING ALGORITHMS
- NUMERICAL INTEGRATION ALGORITHMS
- ORDINARY DIFFERENTIAL EQUATIONS
- PARTIAL DIFFERENTIAL EQUATIONS
"""

########################################################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
########################################################################################################################
########################################################################################################################
"""
BASIC FUNCTIONS

- FACTORIAL - Factorial of a natural number
- SINE - Sine function with Taylor series expansion
- EXP - Exponential function with Taylor series expansion
- ROUND - Round a number to a certain number of decimal places
- round_matrix - Round off all elements of a matrix
- derivative - Find the derivative of a function at a given x
- double_derivative - Find the double derivative of a function at a given x
"""
########################################################################################################################
########################################################################################################################


"""
Factorial of a natural number

Parameters:
- n: Natural number

Returns:
- Factorial of the number
"""

def FACTORIAL(n):
    fact=1
    while n>0:
        fact=fact*n
        n-=1
    return fact


########################################################################################################################


"""
Sine function with Taylor series expansion

Parameters:
- x: Argument of the sine function
- n: Number of terms in the Taylor series expansion

Returns:
- Sine of the argument
"""

def SINE(x,n):
    sum=0
    for i in range(n):  # starting the index with i=1 because factorial of -1 is not defined
        d=(-1)**(i) * x**(2*i+1)/FACTORIAL(2*i+1)  # taylor expansion terms
        sum=sum+d
    return sum


########################################################################################################################


"""
Exponential function with Taylor series expansion

Parameters:
- x: Argument of the exponential function
- n: Number of terms in the Taylor series expansion

Returns:
- Exponential of the argument
"""

def EXP(x,n):
    sum=0
    for i in range(0,n):
        d=(-1)**i * x**i/FACTORIAL(i)  # taylor expansion terms
        sum=sum+d
    return sum


########################################################################################################################


"""
Round a number to a certain number of decimal places.

Parameters:
- n: Number to be rounded
- decimals: Number of decimal places (default = 0)

Returns:
- Rounded number
"""

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def ROUND(n, decimals=10):
    rounded_abs = round_half_up(abs(n), decimals)
    if n>0:
        return rounded_abs
    elif n<0:
        return(-1)*rounded_abs
    else:
        return 0


########################################################################################################################

"""    
Function to round off all elements of a matrix

Parameters:
- M: Matrix

Returns:
- Matrix with all elements rounded off to 2 decimal places
"""

def round_matrix(M):
    for i in range(len(M)):
        for j in range(len(M[0])):
            M[i][j]=ROUND(M[i][j],2)
    return M


########################################################################################################################


"""
Function for finding derivative of a function at given x

Parameters:
- f: Function for which the derivative is to be found
- x: Value at which the derivative is to be found

Returns:
- Derivative of the function at the given x
"""

def derivative(f, x):
    h=10**-8
    dy_dx=(f(x+h)-f(x))/h # Derivative algorithm
    return dy_dx


"""
Function for finding double derivative of a function at given x

Parameters:
- f: Function for which the double derivative is to be found
- x: Value at which the double derivative is to be found

Returns:
- Double derivative of the function at the given x
"""

def double_derivative(f, x, h=1e-4):
    # Calculate the second derivative using finite differences
    d2y_dx2 = (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)
    return d2y_dx2

########################################################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
########################################################################################################################
########################################################################################################################
"""
ROOT FINDING ALGORITHMS

- Bracketing - Root bracketing algorithm
- Bisection - Root finding using bisection method
    - Bisection_for_plotting - Bisection method for plotting convergence
- Regula Falsi - Root finding using Regula Falsi method
    - Regula Falsi_for_plotting - Regula Falsi method for plotting convergence
- Newton Raphson - Root finding using Newton-Raphson method
    - Newton Raphson_for_plotting - Newton-Raphson method for plotting convergence
- Fixed Point Method - Root finding using fixed-point method
- Laguerre - Root finding using Laguerre method
    - poly_function - Polynomial function
    - deflate - Synthetic division or deflation
    - laguerre - Laguerre method of finding roots for polynomial function
    - poly_solution - Collect all the roots and deflate the polynomial
- Secant Method - Root finding using the secant method
"""
########################################################################################################################
########################################################################################################################


"""
Function for bracketing the root 
The algorithm changes the intervals towards lower value among f(a) and f(b)

Parameters:
- f: Function for which the root is to be found
- a: Lower limit of the interval
- b: Upper limit of the interval

Returns:
- Bracketed interval
"""

def bracketing(f, a, b):
    scale=0.1 # defining scaling factor for changing the interval
        
    while f(a)*f(b)>0:
        if abs(f(a)) <= abs(f(b)):
            a = a - scale*(b-a)
        else:
            b = b + scale*(b-a)
    return a, b


########################################################################################################################


"""
Function for finding root using bisection method i.e. c=(a+b)/2

Parameters:
- f: Function for which the root is to be found
- a: Lower limit of the interval
- b: Upper limit of the interval
- tol: Tolerance (default = 1e-6)

Returns:
- Root of the function using bisection method
"""

def bisection(f, a, b, tol=1e-6):
    # Checking if root is landed by default - really lucky
    if f(a)*f(b)==0.0:
        if f(a)==0.0:
            return a
        else:
            return b

    c=(a+b)/2
    while (b-a)/2>tol: # checking if the accuracy is achieved
        c=(a+b)/2
        if (f(a)*f(c))<=0.0: # Check if the root is properly bracketted
            b=c
        else:
            a=c
    return (a+b)/2


########################################################################################################################


"""
Same bisection function but this gives arrays instead of roots for plotting purpose

Parameters:
- f: Function for which the root is to be found
- a: Lower limit of the interval
- b: Upper limit of the interval
- tol: Tolerance (default = 1e-6)

Returns:
- Arrays for plotting the convergence of the root
"""

def bisection_for_plotting(f, a, b, tol=1e-6):
    loop_count=[]
    lc=0
    loop_value=[]
    root_conv=[]

    # Checking if root is landed by default - really lucky
    if f(a)*f(b)==0:
        lc+=1
        loop_count.append(lc)
        loop_value.append(tol)
        root_conv.append(tol)
        if f(a)==0:
            return a
        else:
            return b

    c=(a+b)/2
    while (b-a)/2>tol: # checking if the accuracy is achieved
        lc+=1
        c=(a+b)/2
        if (f(a)*f(c))<=0: # Check if the root is properly bracketted
            b=c
        else:
            a=c
        loop_count.append(lc)
        root_conv.append((b+a)/2)
        loop_value.append(f((b+a)/2))
    return loop_count, loop_value, root_conv


########################################################################################################################


"""
Function for finding root using Regula-Falsi method i.e. c=b-(b-a)*f(b)/(f(b)-f(a))

Parameters:
- f: Function for which the root is to be found
- a: Lower limit of the interval
- b: Upper limit of the interval
- tol: Tolerance (default = 1e-6)

Returns:
- Root of the function using Regula-Falsi method
"""

def regula_falsi(f, a, b, tol=1e-6):
    # Checking if root is landed by default - really lucky
    if f(a)*f(b)==0:
        if f(a)==0:
            return a
        else:
            return b

    c=(b-a)/2
    cn=b-a
    while abs(c-cn)>tol: # checking if the accuracy is achieved
        cn=c
        c=b-(b-a)*f(b)/(f(b)-f(a))
        if (f(a)*f(c))<=0: # Check if the root is properly bracketted
            b=c
        else:
            a=c
    return c



########################################################################################################################


"""
Same Regula Falsi function but this gives arrays instead of roots for plotting purpose

Parameters:
- f: Function for which the root is to be found
- a: Lower limit of the interval
- b: Upper limit of the interval
- tol: Tolerance (default = 1e-6)

Returns:
- Arrays for plotting the convergence of the root
"""

def regula_falsi_for_plotting(f, a, b, tol=1e-6):
    loop_count=[]
    lc=0
    loop_value=[]
    root_conv=[]

    # Checking if root is landed by default - really lucky
    if f(a)*f(b)==0:
        lc+=1
        loop_count.append(lc)
        loop_value.append(tol)
        root_conv.append(tol)
        if f(a)==0:
            return a
        else:
            return b

    c=(b-a)/2
    cn=b-a
    while abs(c-cn)>tol: # checking if the accuracy is achieved
        lc+=1
        cn=c
        c=b-(b-a)*f(b)/(f(b)-f(a))
        if (f(a)*f(c))<=0: # Check if the root is properly bracketted
            b=c
        else:
            a=c
        loop_count.append(lc)
        root_conv.append(c)
        loop_value.append(f(c))
    return loop_count, loop_value, root_conv


########################################################################################################################


"""
# Function for finding root using newton-raphson method i.e. x=x-f(x)/deriv(f,x)
# when given a guess solution x far from extrema

Parameters:
- x: Initial guess for the root
- f: Function for which the root is to be found
- tol: Tolerance (default = 1e-6)
- max_it: Maximum number of iterations (default = 100)

Returns:
- Root of the function using newton-raphson method
"""

def newton_raphson(f, x, tol=1e-6, max_it=100):
    xn=x
    k=0
    x=x-f(x)/derivative(f,x)
    while abs(x-xn)>tol and k<max_it: # checking if the accuracy is achieved
        xn=x
        x=x-f(x)/derivative(f,x)
        k+=1
    return x


########################################################################################################################


"""
Same newton-raphson function but this gives arrays instead of roots for plotting purpose

Parameters:
- x: Initial guess for the root
- f: Function for which the root is to be found
- tol: Tolerance (default = 1e-6)
- max_it: Maximum number of iterations (default = 100)

Returns:
- Arrays for plotting the convergence of the root
"""

def newton_raphson_for_plotting(f, x, tol=1e-6, max_it=100):
    loop_count=[]
    lc=0
    k=0
    loop_value=[]
    root_conv=[]
    xn=x
    x=x-f(x)/derivative(f,x)
    while abs(x-xn)>tol and k<max_it: # checking if the accuracy is achieved
        lc+=1
        xn=x
        k+=1
        x=x-f(x)/derivative(f,x)
        loop_count.append(lc)
        root_conv.append(x)
        loop_value.append(f(x))
    return loop_count, loop_value, root_conv


########################################################################################################################


"""
Root finding using fixed-point method.

Parameters:
- g(x): The function for which we want to find the root
- x0: Initial guess for the root
- tol: Tolerance (default = 1e-6)
- max_iter: Maximum number of iterations (default = 100)

Returns:
- root: Approximate root found by the fixed-point iteration.
- iterations: Number of iterations performed.
"""

def fixed_point_method(g, x0, tol=1e-6, max_iter=100):
    for iterations in range(1, max_iter):
        x1 = g(x0)
        if abs(x1 - x0) < tol:
            return x1, iterations
        x0 = x1
    raise RuntimeError("Fixed-point iteration did not converge within the maximum number of iterations. Try a different initial guess of g(x).")


########################################################################################################################


"""
Functions for laguerre method
"""

"""
Function to give the polynomial P(x) given coefficient array

Parameters:
- A: Coefficient array

Returns:
- Polynomial function
"""

def poly_function(A):
    def p(x):
        n=len(A)
        s=0
        for i in range(n):
            s+=A[i]*x**(n-1-i)
        return s
    return p



"""
Function for synthetic division - deflation
it works simply the sythetic division way, the ouptput coefficients are stored in array C

Parameters:
- A: Coefficient array
- sol: Solution

Returns:
- Coefficient array after deflation
"""

def deflate(A, sol):
    n=len(A)
    B=[0 for i in range(n)]
    C=[0 for i in range(n-1)]
    C[0]=A[0]
    for i in range(n-1):
        B[i+1]=C[i]*sol
        if i!=n-2:
            C[i+1]=A[i+1]+B[i+1]
    return C



"""
Function for laguerre method of finding roots for polynomial function
this functions works only when all roots are real.
may give garbage values if polynomials with complex roots are taken

Parameters:
- A: Coefficient array
- guess: Initial guess for the root
- tol: Tolerance (default = 1e-6)
- max_iter: Maximum number of iterations (default = 100)

Returns:
- Root of the polynomial
"""

def laguerre(A, guess, tol=1e-6, max_iter=100):
    n = len(A)
    #define the polynomial function
    p = poly_function(A)
    #check if guess was correct
    x = guess
    if p(x) == 0:
        return x
    
    # defining a range for max iterations, so that it does not run into infinite loops
    # the functions here must converge in this limit, else it is not a good guess
    for i in range(max_iter):
        xn = x
        
        G = derivative(p,x)/p(x)
        H = G**2 - double_derivative(p,x)/p(x)
        denom1 = G+((n-2)*((n-1)*H - G**2))**0.5
        denom2 = G-((n-2)*((n-1)*H - G**2))**0.5
        
        #compare denominators
        if abs(denom2)>abs(denom1):
            a = (n-1)/denom2
        else:
            a = (n-1)/denom1
        
        x = x-a
        #check if convergence criteria satisfied
        if abs(x-xn) < tol:
            return x
    return x # Change it to return False since it would not converge



"""
Function to collect all the roots and deflate the polynomial

Parameters:
- A: Coefficient array
- x: Initial guess for the root

Returns:
- List of roots of the polynomial
"""

def poly_solution(A, x):
    n = len(A)
    p=poly_function(A)
    roots = []
    
    for i in range(n-1):
        root = laguerre(A, x)
        
        # newton raphson for polishing the roots
        root=newton_raphson(p, root)

        # appending the root into list
        roots.append(root)
        
        # deflating the polynomial by synthetic division
        A = deflate(A, root)
    return roots


########################################################################################################################


"""
Secant Method for finding the root of a function.

Parameters:
    func: The function for which to find the root.
    x0: The first initial guess for the root.
    x1: The second initial guess for the root.
    epsilon: The desired accuracy of the solution.
    max_iter: The maximum number of iterations.

Returns:
    float: The estimated root of the function.
    ValueError: If the maximum number of iterations is reached without convergence.
"""

def secant_method(func, x0, x1, epsilon=1e-8, max_iter=100):
    x_prev = x0
    x_curr = x1

    for _ in range(max_iter):
        f_prev = func(x_prev)
        f_curr = func(x_curr)

        if abs(f_curr) < epsilon:
            return x_curr

        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)

        if abs(x_next - x_curr) < epsilon:
            return x_next

        x_prev = x_curr
        x_curr = x_next

    raise ValueError("Secant method did not converge within the maximum number of iterations.")

########################################################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
########################################################################################################################
########################################################################################################################
"""
NUMERICAL INTEGRATION ALGORITHMS

- Mid-point rule
    - find_max_abs_f_2nd_derivative - Find the maximum value of the absolute value of the 2nd derivative of the function
    - calculate_N_mp - Calculate the number of subintervals required for the Mid-point rule to achieve a certain error tolerance
    - int_mid_point - Numerical integration by mid-point method
- Trapezoidal rule
    - calculate_N_t - Calculate the number of subintervals required for the Trapezoidal rule to achieve a certain error tolerance
    - int_trapezoidal - Numerical integration by Trapezoidal method
- Simpson's rule
    - find_max_abs_f_4th_derivative - Find the maximum value of the absolute value of the 4th derivative of the function
    - calculate_N_s - Calculate the number of subintervals required for the Simpson's rule to achieve a certain error tolerance
    - int_simpson - Numerical integration using the Simpson's rule
- Gaussian or Gauss-Legendre quadrature
    - legendre_polynomial - Legendre polynomial function
    - legendre_derivative - Derivative of Legendre polynomial
    - find_root - Find the roots of Legendre polynomial of order n using Newton's method
    - get_roots_weights_gaussian - Get the roots and weights of the Gaussian quadrature
    - gauss_quad - Gaussian quadrature for a given order of the Legendre polynomial
    - Gaussian_quadrature - Gaussian quadrature for a given function and interval
"""
########################################################################################################################
########################################################################################################################


"""
Find the maximum value of the absolute value of the 2nd derivative of the function.

Parameters:
- f: The function for which we want to find the maximum value of the absolute value of the 2nd derivative
- a: Lower limit of the interval
- b: Upper limit of the interval

Returns:
- Maximum value of the absolute value of the 2nd derivative of the function
"""

def find_max_abs_f_2nd_derivative(f, a, b, *args):
    h = (b - a) / 1000
    x = [a+i*h for i in range(1000)]
    y = []

    for i in range(len(x)):
        # calculate the 2nd derivative of f(x) using the central difference method
        y.append(abs((f(x[i] + h, *args) - 2*f(x[i], *args) + f(x[i] - h, *args)) / h**2))
            
    return max(y)


"""
Calculate the number of subintervals required for the Mid-point rule to achieve a certain error tolerance.

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- tol: Tolerance (default = 1e-6)

Returns:
- N_mp: Number of subintervals for the Mid-point rule
"""

def calculate_N_mp(f, a, b, tol=1e-6, *args):

    fn_mp = find_max_abs_f_2nd_derivative(f, a, b, *args)

    # Calculation of N from error calculation formula
    N_mp=int(((b-a)**3/24/tol*fn_mp)**0.5)
    
    if N_mp==0:
        N_mp=1

    return N_mp


"""
Numerical integration by mid-point method

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- tol: Tolerance (default = 1e-6)

Returns:
- I: Approximate value of the integral by the mid-point method
"""

def int_mid_point(f, a, b, tol=1e-6, *args):
    N = calculate_N_mp(f, a, b, tol, *args)
    s=0
    h=(b-a)/N # step size
    
    # integration algorithm
    for i in range(1,N+1):
        x=a+(2*i-1)*h/2
        s+=f(x)
    
    return s*h


########################################################################################################################


"""
Calculate the number of subintervals required for the Trapezoidal rule to achieve a certain error tolerance.

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- tol: Tolerance (default = 1e-6)

Returns:
- N_t: Number of subintervals for the Trapezoidal rule
"""

def calculate_N_t(f, a, b, tol=1e-6, *args):

    fn_t = find_max_abs_f_2nd_derivative(f, a, b, *args)

    # Calculation of N from error calculation formula
    N_t=int(((b-a)**3/12/tol*fn_t)**0.5)
    
    if N_t==0:
        N_t=1

    return N_t


"""
Numerical integration by Trapezoidal method

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- tol: Tolerance (default = 1e-6)

Returns:
- I: Approximate value of the integral by the Trapezoidal method
"""

def int_trapezoidal(f, a, b, tol=1e-6, *args):
    N = calculate_N_t(f, a, b, tol, *args)
    s=0
    h=(b-a)/N # step size
    
    # integration algorithm
    for i in range(1,N+1):
        s+=f(a+i*h)+f(a+(i-1)*h)
    
    return s*h/2


########################################################################################################################


"""
Find the maximum value of the absolute value of the 4th derivative of the function.

Parameters:
- f: The function for which we want to find the maximum value of the absolute value of the 4th derivative
- a: Lower limit of the interval
- b: Upper limit of the interval

Returns:
- Maximum value of the absolute value of the 4th derivative of the function
"""

def find_max_abs_f_4th_derivative(f, a, b, *args):
    h = (b - a) / 1e3
    x = [a+i*h for i in range(1000)]
    y = []

    for i in range(len(x)):
        # calculate the 4th derivative of f(x) using the central difference method
        y.append(abs((f(x[i] + 2*h, *args) - 4*f(x[i] + h, *args) + 6*f(x[i], *args) - 4*f(x[i] - h, *args) + f(x[i] - 2*h, *args)) / h**4))
    
    return max(y)



"""
Calculate the number of subintervals required for the Simpson's rule to achieve a certain error tolerance.

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- tol: Tolerance (default = 1e-6)

Returns:
- N_s: Number of subintervals
"""

def calculate_N_s(f, a, b, tol=1e-6, *args):

    fn_s = find_max_abs_f_4th_derivative(f, a, b, *args)

    # Calculation of N from error calculation formula
    N_s=int(((b-a)**5/180/tol*fn_s)**0.25)
    
    if N_s==0:
        N_s=2
    
    # Special case with simpson's rule
    # It is observed for simpson rule for even N_s, it uses same value
    # but for odd N_s, it should be 1 more else the value is coming wrong
    if N_s%2!=0:
        N_s+=1

    return N_s



'''
Numerical integration using the Simpson's rule.

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- tol: Tolerance (default = 1e-6)

Returns:
- I: Approximate value of the integral by the Simpson's rule
'''

def int_simpson(f, a, b, tol=1e-8, *args):
    N = calculate_N_s(f, a, b, tol, *args)
    s = f(a, *args) + f(b, *args)
    h = (b - a) / N
    
    # integration algorithm
    for i in range(1, N):
        if i % 2 != 0:
            s += 4 * f(a + i * h, *args)
        else:
            s += 2 * f(a + i * h, *args)
    
    return s * h / 3


########################################################################################################################

# Function to find the roots of Legendre polynomial given order using Newton's method. 
# I have manually calculated the roots and weights here. This was just to show the working of the code. 
# However, all these calculations are not required to do iteratively as the root and weight can be calculated once and saved.

"""
Legendre polynomial function.

Parameters:
- x: Initial guess for the root
- n: Order of the Legendre polynomial

Returns:
- P(x): Legendre polynomial at given x and order n
"""

def legendre_polynomial(x, n):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return ((2 * n - 1) * x * legendre_polynomial(x, n - 1) - (n - 1) * legendre_polynomial(x, n - 2)) / n



"""
Function to find the derivative of Legendre polynomial

Parameters:
- x: Initial guess for the root
- n: Order of the Legendre polynomial

Returns:
- P'(x): Derivative of Legendre polynomial at given x and order n
"""

def legendre_derivative(x, n):
    return n * (x * legendre_polynomial(x, n) - legendre_polynomial(x, n - 1)) / (x**2 - 1)



"""
Function to find the roots of Legendre polynomial of order n using Newton's method.

Parameters:
- initial_guess: Initial guess for the root
- n: Order of the Legendre polynomial

Returns:
- x: Roots of the Legendre polynomial
"""

def find_root(initial_guess, n):
    tolerance = 1e-12
    max_iterations = 1000
    x = initial_guess

    for _ in range(max_iterations):
        f_x = legendre_polynomial(x, n)
        f_prime_x = legendre_derivative(x, n)
        x -= f_x / f_prime_x

        if abs(f_x) < tolerance:
            break

    return x



"""
Function to find the roots and weights of the Gaussian quadrature for a given order of the Legendre polynomial.

Parameters:
- n: Order of the Legendre polynomial

Returns:
- roots: Roots of the Legendre polynomial
- weights: Weights of the Legendre polynomial
"""

def get_roots_weights_gaussian(n):
    guess = [np.cos((2 * i + 1) * np.pi / (2 * n)) for i in range(n)]
    roots = [find_root(guess[i], n) for i in range(n)]
    weights = [2 / ((1 - root**2) * legendre_derivative(root, n)**2) for root in roots]

    return roots, weights


# These 3 functions are not required to be calculated everytime while doing actual calculations as 
# the root and weight calculations can be done once and saved in a file. 
########################################################################################################################


"""
Gaussian quadrature for a given order of the Legendre polynomial.

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- ord: Order of the Legendre polynomial

Returns:
- Gauss_int: Approximate value of the integral
"""

def gauss_quad(f, a, b, ord):
    roots, weights = get_roots_weights_gaussian(ord)
    Gauss_int = 0
    for i in range(ord):
        x_i = 0.5 * (b - a) * roots[i] + 0.5 * (a + b)
        Gauss_int += weights[i] * f(x_i)
    Gauss_int *= 0.5 * (b - a)
   
    return Gauss_int



"""
Gaussian quadrature for a given function and interval.

Parameters:
- f: The function to be integrated
- a: Lower limit of integration
- b: Upper limit of integration
- tol: Tolerance (default = 1e-8)

Returns:
- GQ1: Approximate value of the integral
- ord+1: Order of the Legendre polynomial
"""

def Gaussian_quadrature(f, a, b, tol=1e-8):
    for ord in range(2, 30):
        GQ0 = gauss_quad(f, a, b, ord)
        GQ1 = gauss_quad(f, a, b, ord+1)
        if abs(GQ1 - GQ0) < tol:
            return GQ1, ord

    return ValueError("Integral did not converge within 30 orders of Legendre polynomials.")


########################################################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
########################################################################################################################
########################################################################################################################
"""
MATRICES AND LINEAR ALGEBRA

- print_matrix - Function to print a matrix
- print_matrix_with_gap - Function to print a matrix
- add_matrix - Function to add two matrices
- subtract_matrix - Function to subtract two matrices
- multiply_scalar - Function to multiply a matrix by a scalar
- multiply_matrix - Function to multiply two matrices
- transpose_matrix - Function to transpose a matrix
- read_matrix - Function for reading the matrix from a text file
- swap_rows - Function to swap two rows of a matrix
- partial_pivot - Function for partial pivoting
- gauss_jordan - Function for Gauss-Jordan elimination
- get_inv_GJ - Function to get the inverse of a matrix using Gauss-Jordan elimination
- gauss_jordan_steps - Function to get the steps of Gauss-Jordan elimination
- partial_pivot_LU - Function for partial pivoting in LU decomposition
- determinant_LU - Function to find the determinant of a matrix using LU decomposition
- get_identity - Function to get the identity matrix of a given order
- check_positive_definite - Function to check if a matrix is positive definite
- LU_doolittle - Function for LU decomposition using Doolittle's method
    - for_back_subs_dooolittle - Function for forward and backward substitution using LU decomposition
- LU_crout - Function for LU decomposition using Crout's method
    - for_back_subs_crout - Function for forward and backward substitution using LU decomposition
- inverse_by_LU_decomposition - Function to get the inverse of a matrix using LU decomposition
- LU_cholesky - Function for LU decomposition using Cholesky's method
    - for_back_subs_cholesky - Function for forward and backward substitution using LU decomposition
- LU_do2
- jacobi - Function for Jacobi's method
- gauss_seidel - Function for Gauss-Seidel method
"""
########################################################################################################################
########################################################################################################################


"""
Function to print a matrix

Parameters:
- A: Matrix

Returns:
- Prints the matrix with appropriate spaces for easy understanding
"""


def print_matrix(A):
    r = len(A)
    c = len(A[0])
    for i in range(r):
        for j in range(c):
            # prints the matrix with appropriate spaces for easy understanding
            print(str(A[i][j]).ljust(7), end="")
        print()
    print()



"""
Function to print a matrix

Parameters:
- A: Matrix

Returns:
- Prints the matrix with appropriate spaces for easy understanding
"""


def print_matrix_with_gap(A):
    r = len(A)
    c = len(A[0])
    for i in range(r):
        for j in range(c):
            # prints the matrix with appropriate spaces for easy understanding
            print(A[i][j], end="    ")
        print()
    print()


########################################################################################################################


"""
Function to add two matrices

Parameters:
- A: First matrix
- B: Second matrix

Returns:
- Sum of the two matrices
"""

def add_matrix(A, B):
    r = len(A)
    c = len(A[0])
    C=[[0 for i in range(c)] for j in range(r)]
    for i in range(r):
        for j in range(c):
            C[i][j]=A[i][j]+B[i][j] # Addition algorithm
    return C


########################################################################################################################


"""
Function to subtract two matrices

Parameters:
- A: First matrix
- B: Second matrix

Returns:
- Difference of the two matrices
"""

def subtract_matrix(A, B):
    r = len(A)
    c = len(A[0])
    C=[[0 for i in range(c)] for j in range(r)]
    for i in range(r):
        for j in range(c):
            C[i][j]=A[i][j]-B[i][j] # Subtraction algorithm
    return C


########################################################################################################################


"""
Function to multiply a matrix by a scalar

Parameters:
- A: Matrix
- s: Scalar

Returns:
- Product of the matrix and the scalar
"""

def multiply_scalar(A, s):
    r = len(A)
    c = len(A[0])
    B=[[0 for i in range(c)] for j in range(r)]
    for i in range(r):
        for j in range(c):
            B[i][j]=s*A[i][j] # Multiplication algorithm
    return B


########################################################################################################################


"""
Function to multiply two matrices

Parameters:
- A: First matrix
- r1: Number of rows of the first matrix
- c1: Number of columns of the first matrix
- B: Second matrix
- r2: Number of rows of the second matrix
- c2: Number of columns of the second matrix

Returns:
- Product of the two matrices
"""

def multiply_matrix(A, r1, c1, B, r2, c2):
    if c1==r2: # checking compatibility
        C=[[0 for i in range(c2)] for j in range(r1)] # initializing matrix C
        for i in range(r1):
            for j in range(c2):
                for k in range(c2):
                    C[i][j]+=float(A[i][k])*float(B[k][j]) # multiplication algorithm
        return C, r1, c2
    else:
        print("matrices incompatible for multiplication")


########################################################################################################################


"""
Function to transpose a matrix

Parameters:
- A: Matrix

Returns:
- Transpose of the matrix
"""

def transpose_matrix(A):
    r = len(A)
    c = len(A[0])
    B = [[0 for x in range(r)] for y in range(c)] 
    for i in range(r):
        for j in range(c):
            B[j][i]=A[i][j]
    return B


########################################################################################################################


"""
Function for reading the matrix from a text file

Parameters:
- txt: Text file containing the matrix

Returns:
- Matrix, number of rows, number of columns
"""

def read_matrix(txt):
    with open(txt, 'r') as a:
        matrix=[[float(num) for num in row.split(' ')] for row in a ]
    row=len(matrix)
    column=len(matrix[0])
    return matrix, row, column


########################################################################################################################


"""
Function to swap two rows of a matrix

Parameters:
- A: Matrix
- row1: First row
- row2: Second row

Returns:
- Matrix with row1 and row2 swapped
"""

def swap_rows(A, row1, row2):
    temp = A[row1]
    A[row1] = A[row2]
    A[row2] = temp
    return A



"""
Function for partial pivoting

Parameters:
- Ab: Augmented matrix
- m: Current row
- nrows: Number of rows

Returns:
- Augmented matrix with partial pivoting
"""

def partial_pivot(Ab, m, nrows):
    pivot = Ab[m][m]    # declaring the pivot
    if (Ab[m][m] != 0):
        return Ab    # return if partial pivot is not required
    else:
        for r in range(m+1,nrows):
            pivot=Ab[r][m]
            # check for non-zero pivot and swap rows with it
            for k in range(m+1,nrows):
                if abs(Ab[k][m])>pivot:
                    pivot=Ab[k][m]
                    r=k
            if Ab[r][m] != 0:
                pivot = Ab[r][m]
                Ab=swap_rows(Ab,m,r)
                return Ab
            else:
                r+=1
    if (pivot==0):    # no unique solution case
        return None


########################################################################################################################
    

"""
Gauss Jordan Elimiination method

Parameters:
- Ab: Augmented matrix
- nrows: Number of rows
- ncols: Number of columns

Returns:
- Augmented matrix after Gauss Jordan elimination
"""

def gauss_jordan(Ab,nrows,ncols):
    det=1
    r=0
    # does partial pivoting
    Ab = partial_pivot(Ab,r,nrows)
    for r in range(0,nrows):
        # no solution case
        if Ab==None:
            return Ab
        else:
            # Changes the diagonal elements to unity
            fact=Ab[r][r]
            if fact==0:
                # does partial pivoting
                Ab = partial_pivot(Ab,r,nrows)
            fact=Ab[r][r]
            det=det*fact # calculates the determinant
            for c in range(r,ncols):
                Ab[r][c]*=1/fact
            # Changes the off-diagonal elements to zero
            for r1 in range(0,nrows):
                # does not change if it is already done
                if (r1==r or Ab[r1][r]==0):
                    r1+=1
                else:
                    factor = Ab[r1][r]
                    for c in range(r,ncols):
                        Ab[r1][c]-= factor * Ab[r][c]
    return Ab, det


########################################################################################################################


"""
Function to extract inverse from augmented matrix

Parameters:
- A: Augmented matrix
- n: Number of rows

Returns:
- Inverse matrix
"""

def get_inv_GJ(A,n):
    r=len(A)
    c=len(A[0])
    M=[[0 for j in range(n)] for i in range(n)]
    for i in range(r):
        for j in range(n,c):
            M[i][j-n]=A[i][j]
    return M


########################################################################################################################


"""
Gauss Jordan Elimiination method - for reference to know what happens at each step

Parameters:
- Ab: Augmented matrix
- nrows: Number of rows
- ncols: Number of columns

Returns:
- Augmented matrix after Gauss Jordan elimination
"""


def gauss_jordan_steps(Ab,nrows,ncols):
    # does partial pivoting
    det=1
    r=0
    Ab = partial_pivot(Ab,r,nrows)
    for r in range(0,nrows):
        # no solution case
        if Ab==None:
            return Ab
        else:
            # Changes the diagonal elements to unity
            print("value of  r  =  "+str(r))
            print_matrix(Ab,nrows,ncols)
            fact=Ab[r][r]
            if fact==0:
                # does partial pivoting
                Ab = partial_pivot(Ab,r,nrows)
            fact=Ab[r][r]
            print("changing values of diagonal")
            det=det*fact # calculates the determinant
            for c in range(r,ncols):
                print("fact value  =  "+str(fact))
                Ab[r][c]*=1/fact
                print_matrix(Ab,nrows,ncols)
                print("loop -> value of  c  =  "+str(c))
            # Changes the off-diagonal elements to zero
            print("Now changing values other than diagonal")
            for r1 in range(0,nrows):
                # does not change if it is already done
                print("loop -> value of  r1  =  "+str(r1)+"  when  r  =  "+str(r))
                if (r1==r or Ab[r1][r]==0):
                    r1+=1
                else:
                    factor = Ab[r1][r]
                    for c in range(r,ncols):
                        Ab[r1][c]-= factor * Ab[r][c]
                print_matrix(Ab,nrows,ncols)
    return Ab, det


########################################################################################################################


"""
Function for partial pivot for LU decomposition

Parameters:
- mat: Matrix
- vec: Vector
- n: Number of rows

Returns:
- Matrix and vector after partial pivot
"""

def partial_pivot_LU (mat, vec, n):
    for i in range (n-1):
        if mat[i][i] ==0:
            for j in range (i+1,n):
                # checks for max absolute value and swaps rows 
                # of both the input matrix and the vector as well
                if abs(mat[j][i]) > abs(mat[i][i]):
                    mat[i], mat[j] = mat[j], mat[i]
                    vec[i], vec[j] = vec[j], vec[i]
    return mat, vec


########################################################################################################################


"""
Function to calculate the determinant of a matrix via product of transformed L or U matrix

Parameters:
- mat: Matrix
- n: Number of rows

Returns:
- Determinant of the matrix
"""

def determinant_LU(mat,n):
    det=1
    for i in range(n):
        det*=-1*mat[i][i]
    return det


########################################################################################################################


"""
Function to produce n x n identity matrix

Parameters:
- n: Size of matrix

Returns:
- Identity matrix
"""

def get_identity(n):
    I=[[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        I[i][i]=1
    return I


########################################################################################################################

"""
Function for checking hermitian matrix for cholesky decomposition

Parameters:
- mat: Matrix

Returns:
- True if hermitian, False otherwise
"""

def check_positive_definite(mat):
    l=0
    n=len(mat)
    for i in range(n):
        for j in range(n):
            if mat[i][j]==mat[j][i]:
                l+=1
    if l==n**2:
        return(True)
    else:
        return(False)


########################################################################################################################


"""
LU decomposition using Doolittle's condition L[i][i]=1 without making separate L and U matrices

Parameters:
- mat: Matrix
- n: Number of rows

Returns:
- LU decomposition of the matrix
"""

def LU_doolittle(mat,n):
    for i in range(n):
        for j in range(n):
            if i>0 and i<=j: # changing values of upper triangular matrix
                sum=0
                for k in range(i):
                    sum+=mat[i][k]*mat[k][j]
                mat[i][j]=mat[i][j]-sum
            if i>j: # changing values of lower triangular matrix
                sum=0
                for k in range(j):
                    sum+=mat[i][k]*mat[k][j]
                mat[i][j]=(mat[i][j]-sum)/mat[j][j]
    return mat



"""
Function to find the solution matrix provided a vector using forward and backward substitution respectively

Parameters:
- mat: Matrix
- n: Number of rows
- vect: Vector in RHS

Returns:
- Solution vector
"""

def for_back_subs_doolittle(mat,n,vect):
    # initialization
    y=[0 for i in range(n)]
    
    # forward substitution
    y[0]=vect[0]
    for i in range(n):
        sum=0
        for j in range(i):
            sum+=mat[i][j]*y[j]
        y[i]=vect[i]-sum
    
    # backward substitution
    vect[n-1]=y[n-1]/mat[n-1][n-1]
    for i in range(n-1,-1,-1):
        sum=0
        for j in range(i+1,n):
            sum+=mat[i][j]*vect[j]
        vect[i]=(y[i]-sum)/mat[i][i]
    del(y)
    return vect


########################################################################################################################


"""
LU decomposition using Crout's condition U[i][i]=1 without making separate L and U matrices

Parameters:
- mat: Matrix
- n: Number of rows

Returns:
- LU decomposition of the matrix
"""

def LU_crout(mat,n):
    for i in range(n):
        for j in range(n):
            if i>=j: # changing values of lower triangular matrix
                sum=0
                for k in range(j):
                    sum+=mat[i][k]*mat[k][j]
                mat[i][j]=mat[i][j]-sum
            if i<j: # changing values of uppr triangular matrix
                sum=0
                for k in range(i):
                    sum+=mat[i][k]*mat[k][j]
                mat[i][j]=(mat[i][j]-sum)/mat[i][i]
    return mat



"""
Function to find the solution matrix provided a vector using forward and backward substitution respectively

Parameters:
- mat: Matrix
- n: Number of rows
- vect: Vector in RHS

Returns:
- Solution vector
"""

def for_back_subs_crout(mat,n,vect):
    y=[0 for i in range(n)]
    
    # forward substitution
    y[0]=vect[0]/mat[0][0]
    for i in range(n):
        sum=0
        for j in range(i):
            sum+=mat[i][j]*y[j]
        y[i]=(vect[i]-sum)/mat[i][i]
    
    # backward substitution
    vect[n-1]=y[n-1]
    for i in range(n-1,-1,-1):
        sum=0
        for j in range(i+1,n):
            sum+=mat[i][j]*vect[j]
        vect[i]=y[i]-sum
    del(y)
    return vect


########################################################################################################################


"""
Find the solution matrix using forward and backward substitution by LU decomposition

Parameters:
- mat: Matrix
- n: Number of rows

Returns:
- Solution vector
"""

def inverse_by_lu_decomposition (matrix, n):

    identity=get_identity(n)
    x=[]
    
    '''
    The inverse finding process could have been done using 
    a loop for the four columns. But while partial pivoting, 
    the rows of final inverse matrix and the vector both are 
    also interchanged. So it is done manually for each row and vector.
    
    deepcopy() is used so that the original matrix doesn't change on 
    changing the copied entities. We reuire the original multiple times here
    
    1. First the matrix is deepcopied.
    2. Then partial pivoting is done for both matrix and vector.
    3. Then the decomposition algorithm is applied.
    4. Then solution is obtained.
    5. And finally it is appended to a separate matrix to get the inverse.
    Note: The final answer is also deepcopied because there is some error 
        due to which all x0, x1, x2 and x3 are also getting falsely appended.
    '''
    
    matrix_0 = copy.deepcopy(matrix)
    partial_pivot_LU(matrix_0, identity[0], n)
    matrix_0 = LU_doolittle(matrix_0, n)
    x0 = for_back_subs_doolittle(matrix_0, n, identity[0])
    x.append(copy.deepcopy(x0))


    matrix_1 = copy.deepcopy(matrix)
    partial_pivot_LU(matrix_1, identity[1], n)
    matrix_1 = LU_doolittle(matrix_1, n)
    x1 = for_back_subs_doolittle(matrix_1, n, identity[1])
    x.append(copy.deepcopy(x1))

    matrix_2 = copy.deepcopy(matrix)
    partial_pivot_LU(matrix_2, identity[2], n)
    matrix_2 = LU_doolittle(matrix_2, n)
    x2 = for_back_subs_doolittle(matrix_2, n, identity[2])
    x.append(copy.deepcopy(x2))

    matrix_3 = copy.deepcopy(matrix)
    partial_pivot_LU(matrix_3, identity[3], n)
    matrix_3 = LU_doolittle(matrix_3, n)
    x3 = for_back_subs_doolittle(matrix_3, n, identity[3])
    x.append(copy.deepcopy(x3))
    
    # The x matrix to be transposed to get the inverse in desired form
    inverse = transpose_matrix(x)
    return (inverse)


########################################################################################################################


"""
Function for Cholesky decomposition
Only works for Hermitian and positive definite matrices. In this case, we use real matrices only

Parameters:
- mat: Matrix
- n: Number of rows

Returns:
- Cholesky decomposition of the matrix
"""

def LU_cho(mat,n):
    if check_positive_definite(mat)==True:
        for i in range(n):
            for j in range(i,n):
                if i==j: # changing diagonal elements
                    sum=0
                    for k in range(i):
                        sum+=mat[i][k]**2
                    mat[i][i]=math.sqrt(mat[i][i]-sum)
                if i<j: # changing upper traiangular matrix
                    sum=0
                    for k in range(i):
                        sum+=mat[i][k]*mat[k][j]
                    mat[i][j]=(mat[i][j]-sum)/mat[i][i]
                    
                    # setting the lower triangular elements same as elements at the transposition
                    mat[j][i]=mat[i][j]
        return mat
    else:
        print("Given matrix is not hermitian, cholesky method cannot be applied.")
        return False


"""
Function to find the solution matrix provided a vector using forward and backward substitution respectively

Parameters:
- mat: Matrix
- n: Number of rows
- vect: Vector in RHS

Returns:
- Solution vector
"""

def for_back_subs_cho(mat,n,vect):
    y=[0 for i in range(n)]
    
    # forward substitution
    y[0]=vect[0]/mat[0][0]
    for i in range(n):
        sum=0
        for j in range(i):
            sum+=mat[i][j]*y[j]
        y[i]=(vect[i]-sum)/mat[i][i]
    
    # forward substitution
    vect[n-1]=y[n-1]
    for i in range(n-1,-1,-1):
        sum=0
        for j in range(i+1,n):
            sum+=mat[i][j]*vect[j]
        vect[i]=(y[i]-sum)/mat[i][i]
    del(y)
    return vect


########################################################################################################################


"""
LU decomposition using Doolittle's condition L[i][i]=1 by making separate L and U matrices

Parameters:
- M: Matrix
- n: Number of rows

Returns:
- M: Matrix with L and U
"""

def LU_do2(M,n):
    # initialization
    L=[[0 for j in range(n)] for i in range(n)]
    U=[[0 for j in range(n)] for i in range(n)]

    for i in range(n):
        L[i][i]=1
        for j in range(n):
            if i>j: 
                U[i][j]=0
            elif i<j:
                L[i][j]=0
            U[0][j]=M[0][j]
            L[i][0]=M[i][0]/U[0][0]
            if i>0 and i<=j: # changing values for upper traiangular matrix
                sum=0
                for k in range(i):
                    sum+=L[i][k]*U[k][j]
                U[i][j]=M[i][j]-sum
            if i>j: # changing values for lower traiangular matrix
                sum=0
                for k in range(j):
                    sum+=L[i][k]*U[k][j]
                L[i][j]=(M[i][j]-sum)/U[j][j]
    print_matrix(L,n,n)
    print_matrix(U,n,n)

    # To check if the L and U matrices are correct, use this for verification
    m,r,c=multiply_matrix(L, n, n, U, n, n)
    print_matrix(m,r,c)
    
    return M


########################################################################################################################


"""
Find the solution of a matrix vector pair using the Jacobi iterative method.
This method is valid only for diagonally dominant matrices.

Parameters:
- A: Matrix
- b: RHS vector
- tol: Tolerance for convergence

Returns:
- x: Solution of the matrix vector pair
"""

def jacobi(matrix, b, prec=1e-4):
    a = 1
    aarr = []
    karr = []
    p = 1
    X = [0] * len(matrix)
    X1 = [0] * len(matrix)

    while a > prec:
        a = 0
        for l in range(len(X)):
            X[l] = X1[l]

        for i in range(len(matrix)):
            Sum = 0
            for j in range(len(matrix)):
                if i != j:
                    Sum += matrix[i][j] * X[j]

            X1[i] = (b[i] - Sum) / matrix[i][i]

        for j in range(len(X)):
            a += (X1[j] - X[j]) ** 2

        a = a ** 0.5
        aarr.append(a)
        karr.append(p)
        p += 1

    return X1


########################################################################################################################


"""
Find the solution of a matrix vector pair using the Gauss-Seidel iterative method.
This method is valid only for diagonally dominant matrices.

Parameters:
- A: Matrix
- b: RHS vector
- tol: Tolerance for convergence

Returns:
- x: Solution of the matrix vector pair
"""

def gauss_seidel(matrix, b, tol=1e-6):
    a = 1
    aarr = []
    karr = []
    p = 1
    X = [0] * len(matrix)
    X1 = [0] * len(matrix)

    while a > tol:
        a = 0
        for l in range(len(X)):
            X1[l] = X[l]

        for i in range(len(matrix)):
            Sum = 0
            for j in range(len(matrix)):
                if i != j:
                    Sum += matrix[i][j] * X[j]

            X[i] = (b[i] - Sum) / matrix[i][i]

        for j in range(len(X)):
            a += (X1[j] - X[j]) ** 2

        a = a ** 0.5
        aarr.append(a)
        karr.append(p)
        p += 1

    return X


########################################################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
########################################################################################################################
########################################################################################################################
"""
ORDINARY DIFFERENTIAL EQUATIONS

- forward_euler - Function to solve first order ODE using Forward Euler's method
- backward_euler - Function to solve first order ODE using Backward Euler's method
- predictor_corrector - Function to solve first order ODE using Predictor-Corrector method
- ODE_1D_RK2 - Function to solve first order ODE using Runge-Kutta 2nd order method
- ODE_1ord_RK4 - Function to solve first order ODE using Runge-Kutta 4th order method
- ODE_2ord_RK4 - Function to solve second order ODE using Runge-Kutta 4th order method
- Shooting_method - Function to solve second order ODE using Shooting method
    - ODE_2ord_RK4_for_shooting - Function to solve second order ODE using Runge-Kutta 4th order method
    - Lagrange_interpolation - Function to perform Lagrange interpolation
    - RK4_2ord_shooting_method - Function to solve second order ODE using Runge-Kutta 4th order method
- verlet - Function to solve second order ODE using Verlet method
- velocity_verlet - Function to solve second order ODE using Velocity Verlet method
- leap_frag - Function to solve second order ODE using Leapfrog method
- Simpletic_Euler - Function to solve second order ODE using Simpletic Euler method
- Semi_implicit_Euler - Function to solve second order ODE using Semi-implicit Euler method
"""
########################################################################################################################
########################################################################################################################


"""
Function to solve first order ODE using Forward Euler's method

Parameters:
- x: Initial value of x
- y: Initial value of y
- h: Step size
- lim: Upper limit of x
- dydx: Function for the first derivative of y with respect to x

Returns:
- X: Array of x values
- Y: Array of y values
"""

def forward_euler(x, y, h, lim, dydx):
    # Constructing solution arrays
    X = [x]
    Y = [y]
    
    while x <= lim:
        k1 = h* dydx(x, y) # k1 calculation
        y = y + k1
        x = x + h
        X.append(x)
        Y.append(y)
    return X, Y


########################################################################################################################


"""
Function to solve first order ODE using Backward Euler's method

Parameters:
- f: Function representing the differential equation dy/dt = f(t, y).
- y0: Initial value of the dependent variable.
- x0: Initial value of the independent variable.
- h: Step size.
- lim: Upper limit of the independent variable.

Returns:
- x_values: List of time values.
- y_values: List of corresponding dependent variable values.
"""

def backward_euler(f, y0, x0, h, lim=10):
    N = int(lim/h)
    X = [x0]
    Y = [y0]
    x_n = x0
    y_n = y0

    for i in range(N):
        def func(y_n1):
            return(y_n + h*f(y_n1,x_n+h) - y_n1)
    
        y_nR = newton_raphson(func, x_n)
        y_n1 = y_n + h*f(y_nR,x_n+h)
        X.append(x_n+h)
        Y.append(y_n1)
        y_n = y_n1
        x_n = x_n + h
    return X, Y

########################################################################################################################


"""
Function to solve first order ODE using Predictor-Corrector method

Parameters:
- x: Initial value of x
- y: Initial value of y
- h: Step size
- lim: Upper limit of x
- dydx: Function for the first derivative of y with respect to x

Returns:
- X: Array of x values
- Y: Array of y values
"""

def predictor_corrector(x,y,h, lim, dydx):
    # Constructing solution arrays
    X = [x]
    Y = [y]
    while x <= lim:
        k1 = h* dydx(x, y) # k1 calculation
        k = h* dydx(x+h, y+k1) # k' calculation
        y = y + (k1+k)/2
        x = x + h
        X.append(x)
        Y.append(y)
    return X, Y


########################################################################################################################


"""
Function to solve first order ODE using Runge-Kutta 2nd order method

Parameters:
- x: Initial value of x
- y: Initial value of y
- h: Step size
- lim: Upper limit of x
- dydx: Function for the first derivative of y with respect to x

Returns:
- X: Array of x values
- Y: Array of y values
"""

def ODE_1D_RK2(x,y,h, lim, dydx):
    # Constructing solution arrays
    X = [x]
    Y = [y]
    while x <= lim:
        k1 = h* dydx(x, y) # k1 calculation
        k2 = h* dydx(x+h/2, y+k1/2) # k2 calculation
        y = y + k2
        x = x + h
        X.append(x)
        Y.append(y)
    return X, Y


########################################################################################################################


"""
Function to solve first order ODE using Runge-Kutta 4th order method

Parameters:
- func: Function representing the differential equation dy/dt = func(t, y).
- y0: Initial value of the dependent variable.
- t0: Initial value of the independent variable.
- tn: Final value of the independent variable.
- h: Step size.

Returns:
- x_values: List of time values.
- y_values: List of corresponding dependent variable values.
"""

def ODE_1ord_RK4(dy_dx, y0, x0, xn, h):
    x = [x0]
    y = [y0]

    while x0 < xn:
        k1 = h * dy_dx(x0, y0)
        k2 = h * dy_dx(x0 + 0.5 * h, y0 + 0.5 * k1)
        k3 = h * dy_dx(x0 + 0.5 * h, y0 + 0.5 * k2)
        k4 = h * dy_dx(x0 + h, y0 + k3)

        y0 = y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        x0 = x0 + h

        x.append(x0)
        y.append(y0)

    return x, y


########################################################################################################################


"""
Function to solve second order ODE using Runge-Kutta 4th order method

Parameters:
- x: Initial value of x
- y: Initial value of y
- p: Initial value of dydx
- h: Step size
- l_bound: Lower limit of x
- u_bound: Upper limit of x
- dydx: Function for the first derivative of y with respect to x
- d2ydx2: Function for the second derivative of y with respect to x

Returns:
- X: Array of x values
- Y: Array of y values
- P: Array of p values
"""

def ODE_2ord_RK4(x,y,p, h, l_bound, u_bound, dydx, d2ydx2):
    # p = dy/dx
    x1=x
    y1=y
    p1=p
    
    X=[x]
    Y=[y]
    P=[p]
    while x <= u_bound:
        
        # Calculation for each stepsize h
        k1 = h* dydx(x,y,p)
        l1 = h* d2ydx2(x,y,p)

        k2 = h* dydx(x+h/2, y+k1/2, p+l1/2)
        l2 = h* d2ydx2(x+h/2, y+k1/2, p+l1/2)

        k3 = h* dydx(x+h/2, y+k2/2, p+l2/2)
        l3 = h* d2ydx2(x+h/2, y+k2/2, p+l2/2)

        k4 = h* dydx(x+h, y+k3, p+l3)
        l4 = h* d2ydx2(x+h, y+k3, p+l3)

        y = y + 1/6* (k1 +2*k2 +2*k3 +k4)
        p = p + 1/6* (l1 +2*l2 +2*l3 +l4)
        x = x + h

        # Appending to arrays
        X.append(ROUND(x,8))
        Y.append(ROUND(y,8))
        P.append(ROUND(p,8))

    while x1 >= l_bound:
        
        # Calculation for each stepsize h
        k1 = h* dydx(x,y,p1)
        l1 = h* d2ydx2(x1,y1,p1)

        k2 = h* dydx(x1-h/2, y1-k1/2, p1-l1/2)
        l2 = h* d2ydx2(x1-h/2, y1-k1/2, p1-l1/2)

        k3 = h* dydx(x1-h/2, y1-k2/2, p1-l2/2)
        l3 = h* d2ydx2(x1-h/2, y1-k2/2, p1-l2/2)

        k4 = h* dydx(x1-h, y1-k3, p1-l3)
        l4 = h* d2ydx2(x1-h, y1-k3, p1-l3)

        y1 = y1 - 1/6* (k1 +2*k2 +2*k3 +k4)
        p1 = p1 - 1/6* (l1 +2*l2 +2*l3 +l4)
        x1 = x1-h

        # Appending to arrays
        X.append(ROUND(x1,8))
        Y.append(ROUND(y1,8))
        P.append(ROUND(p1,8))
    return X,Y,P


########################################################################################################################


"""
Function to solve second order ODE using Runge-Kutta 4th order for RK Shooting method

Parameters:
- d2ydx2: Function for the second derivative of y with respect to x
- dydx: Function for the first derivative of y with respect to x
- x0: Initial value of x
- y0: Initial value of y
- z0: Initial value of z

Returns:
- x: Array of x values
- y: Array of y values
- z: Array of z values
"""

def ODE_2ord_RK4_for_shooting(d2ydx2, dydx, x0, y0, z0, xf, h):
    # Yields solution from x=x0 to x=xf
    # y(x0) = y0 & y'(x0) = z0

    # Creating and initialising arrays
    x = []
    x.append(x0)
    y = []
    y.append(y0)
    z = []
    z.append(z0)

    n = int((xf-x0)/h)      # no. of steps
    for i in range(n):
        
        x.append(x[i] + h)
        
        # Calculation for each stepsize h

        k1 = h * dydx(x[i], y[i], z[i])
        l1 = h * d2ydx2(x[i], y[i], z[i])
        
        k2 = h * dydx(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)
        l2 = h * d2ydx2(x[i] + h/2, y[i] + k1/2, z[i] + l1/2)
        
        k3 = h * dydx(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)
        l3 = h * d2ydx2(x[i] + h/2, y[i] + k2/2, z[i] + l2/2)
        
        k4 = h * dydx(x[i] + h, y[i] + k3, z[i] + l3)
        l4 = h * d2ydx2(x[i] + h, y[i] + k3, z[i] + l3)

        y.append(y[i] + (k1 + 2*k2 + 2*k3 + k4)/6)
        z.append(z[i] + (l1 + 2*l2 + 2*l3 + l4)/6)

    return x, y, z



"""
Function for Lagrange's interpolation formula

Parameters:
- chi_h: Higher value of chi
- chi_l: Lower value of chi
- yh: Higher value of y
- yl: Lower value of y
- y: Value of y

Returns:
- chi: Value of chi
"""

def lagrange_interpolation(chi_h, chi_l, yh, yl, y):
    chi = chi_l + (chi_h - chi_l) * (y - yl)/(yh - yl)
    return chi



"""
Solves 2nd order ODE using RK Shooting method with the given boundary conditions

Parameters:
- d2ydx2: Function for the second derivative of y with respect to x
- dydx: Function for the first derivative of y with respect to x
- x_init: Initial value of x
- y_init: Initial value of y
- x_fin: Final value of x

Returns:
- x: Array of x values
- y: Array of y values
- z: Array of z values
"""

def RK4_2ord_shooting_method(d2ydx2, dydx, x_init, y_init, x_fin, y_fin, z_guess1, z_guess2, step_size, tol=1e-6):

    x, y, z = ODE_2ord_RK4_for_shooting(d2ydx2, dydx, x_init, y_init, z_guess1, x_fin, step_size)
    yn = y[-1]

    if abs(yn - y_fin) > tol:
        if yn < y_fin:
            chi_l = z_guess1
            yl = yn

            x, y, z = ODE_2ord_RK4_for_shooting(d2ydx2, dydx, x_init, y_init, z_guess2, x_fin, step_size)
            yn = y[-1]

            if yn > y_fin:
                chi_h = z_guess2
                yh = yn

                # calculate chi using Lagrange interpolation
                chi = lagrange_interpolation(chi_h, chi_l, yh, yl, y_fin)

                # using this chi to solve using RK4
                x, y, z = ODE_2ord_RK4_for_shooting(d2ydx2, dydx, x_init, y_init, chi, x_fin, step_size)
                return x, y, z

            else:
                print("Bracketing FAIL! Try another set of guesses.")


        elif yn > y_fin:
            chi_h = z_guess1
            yh = yn

            x, y, z = ODE_2ord_RK4_for_shooting(d2ydx2, dydx, x_init, y_init, z_guess2, x_fin, step_size)
            yn = y[-1]

            if yn < y_fin:
                chi_l = z_guess2
                yl = yn

                # calculate chi using Lagrange interpolation
                chi = lagrange_interpolation(chi_h, chi_l, yh, yl, y_fin)

                x, y, z = ODE_2ord_RK4_for_shooting(d2ydx2, dydx, x_init, y_init, chi, x_fin, step_size)
                return x, y, z

            else:
                print("GUESSES FAILED! Try another set.")

    else:
        return x, y, z


########################################################################################################################


"""
Function to solve ODE using Verlet method

Parameters:
- A: Function for the acceleration
- x0: Initial value of x
- v0: Initial value of v
- dt: Time step size
- n: Number of time steps
- t0: Initial value of time

Returns:
- T: Array of time values
- X: Array of x values
"""

def verlet(A, x0, v0, dt, n, t0=0):
    # Initialize lists to store positions and time values
    X = [x0]
    T = np.linspace(t0, t0 + dt * n, num=n)

    # Calculate the second position using the initial conditions
    X.append(x0 + v0 * dt + 0.5 * A(x0) * dt**2)

    # Use the Verlet integration method to calculate positions for the remaining time steps
    for i in range(n - 2):
        X.append(2 * X[-1] - X[-2] + A(X[-1]) * dt**2)

    return T, X


########################################################################################################################


"""
Function to solve ODE using Velocity Verlet method

Parameters:
- A: Function for the acceleration
- x0: Initial value of x
- v0: Initial value of v
- dt: Time step size
- n: Number of time steps
- t0: Initial value of time

Returns:
- T: Array of time values
- X: Array of x values
- V: Array of v values
"""

def velocity_verlet(A, x0, v0, dt, n, t0=0):
    # Initialize lists to store positions, velocities, and time values
    X = [x0]
    V = [v0]
    T = np.linspace(t0, t0 + dt * n, num=n)

    for i in range(n - 1):
        # Update positions using current velocity and acceleration
        x_new = X[-1] + V[-1] * dt + 0.5 * A(X[-1]) * dt**2

        # Update velocities using current and next acceleration
        v_new = V[-1] + 0.5 * (A(X[-1]) + A(x_new)) * dt

        # Append new position and velocity to the lists
        X.append(x_new)
        V.append(v_new)

    return T, X, V


########################################################################################################################


"""
Function to solve ODE using Leapfrog method

Parameters:
- F: Function for the force
- x0: Initial value of x
- p0: Initial value of p
- dt: Time step size
- tau: Final value of time
- t0: Initial value of time

Returns:
- Tx: Array of time values for positions
- Tp: Array of time values for momenta
- X: Array of x values
- P: Array of p values
- pf: Final value of p
"""

def leap_frog(F, x0, p0, dt, tau, t0=0):
    # Initialize lists to store positions, momenta, and time values
    X = [x0]
    P = [p0]
    
    # Calculate the number of time steps
    n = int((tau - t0) / dt)
    
    # Generate time values for positions (Tx) and momenta (Tp)
    Tx = [t0 + i * (tau - t0) / (n - 1) for i in range(n)]
    Tp = [t + 0.5 * dt for t in Tx]

    # Calculate the initial momentum using the given force at t0
    P.append(P[-1] + 0.5 * F(t0) * dt)
    
    # Leapfrog integration loop
    for i in range(1, n - 1):
        X.append(X[-1] + P[-1] * dt)
        P.append(P[-1] + F(t0 + i * dt) * dt)
    
    # Final position update
    X.append(X[-1] + P[-1] * dt)
    
    # Final momentum update using force at tau - 0.5 * dt
    pf = P[-1] + F(tau - 0.5 * dt) * 0.5 * dt
    
    return Tx, Tp, X, P, pf


########################################################################################################################


"""
Function for dot product

Parameters:
- a: Vector a
- b: Vector b

Returns:
- Dot product of a and b
"""

def dot_product(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

"""
Function to solve for Q and P using the symplectic Euler method
In this, the Hamiltonian is conserved.

Parameters:
- hamiltonian: Function for the Hamiltonian
- gradient_hamiltonian: Function for the gradient of the Hamiltonian
- q0: Initial value of q
- p0: Initial value of p
- num_steps: Number of time steps
- step_size: Time step size

Returns:
- q_values: Array of q values
- p_values: Array of p values
"""

def symplectic_euler(hamiltonian, gradient_hamiltonian, q0, p0, num_steps, step_size):
    q_values = [[0.0] * len(q0) for _ in range(num_steps + 1)]
    p_values = [[0.0] * len(p0) for _ in range(num_steps + 1)]

    q_values[0] = q0[:]
    p_values[0] = p0[:]

    for i in range(num_steps):
        p_values[i + 1] = [pi - step_size * ghi for pi, ghi in zip(p_values[i], gradient_hamiltonian(q_values[i]))]
        q_values[i + 1] = [qi + step_size * pi1 for qi, pi1 in zip(q_values[i], p_values[i + 1])]

    return q_values, p_values


########################################################################################################################


"""
Function to solve for Q and P using the semi-implicit Euler method

Parameters:
- f1: Function for the first derivative of q
- f2: Function for the first derivative of p
- x0: Initial value of x
- y0: Initial value of y
- dt: Time step size
- num_steps: Number of time steps
- t0: Initial value of time

Returns:
- time_values: Array of time values
- X: Array of x values
- Y: Array of y values
"""

def semi_implicit_euler(f1, f2, x0, y0, dt, num_steps, t0=0):
    X = []
    Y = []
    x = x0
    y = y0
    time_values = np.arange(t0, t0 + num_steps * dt, dt)

    for i in range(num_steps):
        X.append(x)
        Y.append(y)
        y += f2(t0 + i * dt, x) * dt
        x += f1(t0 + i * dt, y) * dt

    return time_values, X, Y


########################################################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
########################################################################################################################
########################################################################################################################
"""
PARTIAL DIFFERENTIAL EQUATIONS

- plot_3D_surface - Function to plot 3D surface plot
- get_matrix_heat_diff - Get the matrices A and B for solving the heat diffusion equation using Crank-Nicolson method
- crank_nicolson_heat_diffusion - Solve 1D heat diffusion equation using Crank-Nicolson method
- poisson_solver - Solve the Poisson equation using implicit finite difference method
- poisson_thomas_solver - Solve the Poisson equation using Thomas algorithm
"""
########################################################################################################################
########################################################################################################################


"""
Function to plot 3D surface plot

Parameters:
- X: Array of x values
- Y: Array of y values
- Sol: Array of solution values

Returns:
- None (Plots the 3D surface plot)
"""

def plot_3D_surface(X, Y, Sol, Title='3D surface plot', X_label='X', Y_label='Y', Z_label='Solution', colormap='plasma', Size=(8, 6)): 
    fig = plt.figure(figsize=Size)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Sol, cmap=colormap)
    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    ax.set_zlabel(Z_label)
    ax.set_title(Title)
    return None


########################################################################################################################
    

"""
Get the matrices A and B for solving the heat diffusion equation using Crank-Nicolson method.

Parameters:
- N: Number of spatial grid points
- sigma: alpha*dt/dx^2

Returns:
- A: Matrix A
- B: Matrix B
"""

def get_matrix_heat_diff(N, sigma):
    A = [[0 for j in range(N)] for k in range(N)]
    B = [[0 for j in range(N)] for k in range(N)]

    for i in range(0, N):
        A[i][i] = 2 + 2*sigma
        B[i][i] = 2 - 2*sigma
        if i > 0:
            A[i][i-1] = -sigma
            B[i][i-1] = sigma
        if i < N-1:
            A[i][i+1] = -sigma
            B[i][i+1] = sigma

    return A, B


"""
Solve 1D heat diffusion equation using Crank-Nicolson method.

Parameters:
- L: Length of the rod
- T: Total time
- dx: Spatial step size
- dt: Time step size
- Diff: Thermal diffusivity

Returns:
- u: Temperature distribution over space and time
- x: Spatial grid
- t: Time grid
"""

def crank_nicolson_heat_diffusion(L, T, dx, dt, Diff, init_cond):

    alpha = Diff * dt / (dx**2)

    # Spatial grid
    x = [i*dx for i in range(int(L/dx)+1)]
    t = [j*dt for j in range(int(T/dt)+1)]

    # Initialize temperature array
    Temp = [[0 for j in range(int(T/dt)+1)] for i in range(len(x))]

    # Initial condition
    for i in range(len(x)):
        Temp[i][0] = init_cond(x[i])

    # Get the matrices for solving the matrix using crank-nicolson method
    A, B = get_matrix_heat_diff(len(x), alpha)

    Temp = np.array(Temp)
    A = np.array(A)
    B = np.array(B)

    for j in range(1, int(T/dt)+1):
        Temp[:, j] = np.linalg.solve(A, np.dot(B, Temp[:, j - 1]))

    return Temp, x, t


########################################################################################################################


"""
Solve the Poisson equation using implicit finite difference method.

Parameters:
- xa: Lower limit of x
- xb: Upper limit of x
- ya: Lower limit of y
- yb: Upper limit of y
- n: Number of grid points in x and y directions
- func_left_bound: Function for the left boundary condition
- func_right_bound: Function for the right boundary condition
- func_bottom_bound: Function for the bottom boundary condition
- func_top_bound: Function for the top boundary condition
- source_func: Function for the source term

Returns:
- x0: Spatial grid in x-direction
- y0: Spatial grid in y-direction
- Sol: Solution of the Poisson equation
"""

def poisson_solver(xa, xb, ya, yb, n, func_left_bound, func_right_bound, func_bottom_bound, func_top_bound, source_func):

    xb += 0.0001*xb
    yb += 0.0001*yb

    # Generate grid
    x0 = np.linspace(xa, xb, num=n)
    y0 = np.linspace(ya, yb, num=n)
    h = (yb - ya) / (n - 1)  # Calculate h_y
    
    # Initialize matrix W
    Sol = np.zeros((n, n))
    
    # Calculate alpha
    alpha = ((xb - xa) / (yb - ya))**2

    # Set boundary conditions
    Sol[0] = func_bottom_bound(x0)
    Sol[n-1] = func_top_bound(x0)

    for i in range(n):
        Sol[i][0] = func_left_bound(y0[i])
        Sol[i][n-1] = func_right_bound(y0[i])

    n2 = n - 2
    R = np.zeros((n2, n2))

    # Populate matrix R with rho values
    for i in range(n2):
        for j in range(n2):
            R[i][j] = -alpha * source_func(x0[i+1], y0[j+1]) * h**2

    R = np.transpose(R)

    N2 = n2**2
    B = np.zeros((n2, n2))

    # Get contributions from boundary conditions
    B[0] = Sol[0][1:-1]
    B[n2-1] = Sol[n-1][1:-1]

    for i in range(n2):
        B[i][0] += alpha * Sol[i+1][0]
        B[i][n2-1] += alpha * Sol[i+1][n-1]

    # Prepare (n-2)*(n-2) dim matrix
    A = np.diag(np.full(N2, 2 + 2*alpha)) - alpha * \
        np.diag(np.ones(N2-1), 1) - alpha * np.diag(np.ones(N2-1), -1) - \
        np.diag(np.ones(N2-n2), n2) - np.diag(np.ones(N2-n2), -n2)

    for i in range(1, n2):
        A[n2*i-1][n2*i] = 0
        A[n2*i][n2*i-1] = 0

    # Invert matrix A
    Ainv = np.linalg.inv(A)

    # Get dot product
    rho = B - R
    u = np.dot(Ainv, rho.flatten())

    # Reshape result to (n-2) x (n-2) matrix
    matrix = u.reshape((n2, n2))

    # Update W with the (x, y) table
    for i in range(n2):
        for j in range(n2):
            Sol[i+1][j+1] = matrix[i][j]

    return x0, y0, Sol


########################################################################################################################


"""
Solve the Poisson equation using Jacobi iterative method also known as Thomas algorithm.

Parameters:
- n_x: Number of grid points in x-direction
- n_y: Number of grid points in y-direction
- x_length: Length of the domain in x-direction
- y_length: Length of the domain in y-direction
- get_BC_poisson: Function to get the boundary conditions

Returns:
- x: Spatial grid in x-direction
- y: Spatial grid in y-direction
- u: Solution of the Poisson equation
"""

def poisson_thomas_solver(n_x, n_y, x_length, y_length, get_BC_poisson):

    n_x += 1
    n_y += 1

    # Discretization
    dx = x_length / (n_x)
    dy = y_length / (n_y)

    # Initialize grid and boundary conditions
    x = [ i*dx for i in range(n_x)]
    y = [ i*dy for i in range(n_y)]

    u = get_BC_poisson(n_x, n_y, x, y)

    # Source term
    src = [[x[i] * math.exp(y[j]) for j in range(n_y)] for i in range(n_x)]

    # Jacobi iterative method
    for _ in range(1000):
        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                u[i][j] = (u[i - 1][j] + u[i][j - 1] + u[i][j + 1] + u[i + 1][j] - dx * dy * src[i][j]) / 4
    
    return np.array(x), np.array(y), np.array(u).T


########################################################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
########################################################################################################################
########################################################################################################################
"""
MONTE CARLO SIMULATIONS

"""
########################################################################################################################
########################################################################################################################


def pdf(x, a, b):
    return 1/(b-a)

def Average(x):
    s=0
    for i in range(len(x)):
        s+=x[i]
    return s/len(x)


def int_monte_carlo(f, pdf, a, b, n):
    I=[]
    for i in range(100):
        x = np.random.uniform(low=a, high=b, size=n)
        F=0
        for i in range(len(x)):
            F+=f(x[i])/pdf(x[i], a, b)
        I.append(F/n)
    return Average(I)


########################################################################################################################



def stdev_s(a):
    sig=0
    mean=0
    for i in range(len(a)):
        mean+=a[i]
    mean=mean/len(a)
    for i in range(len(a)):
        sig+=(a[i]-mean)**2
    sig=math.sqrt(sig/(len(a)-1))
    return sig



def int_monte_carlo_square(f, p, a, b, n):
    x = np.random.uniform(low=a, high=b, size=(n))
    F=0
    for i in range(len(x)):
        F+=(f(x[i]))**2/p(x[i])
    F=F/n
    return F
    


########################################################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
########################################################################################################################
########################################################################################################################
"""
CURVE FITTING ALGORITHMS

"""
########################################################################################################################
########################################################################################################################


# Function to import data from csv file and append to array

def get_from_csv(file):
    C=np.genfromtxt(file, delimiter=',')
    X=[]
    Y=[]
    for i in range(len(C)):
        X.append(C[i][0])
        Y.append(C[i][1])
    return X,Y



# All find statistics

def find_stats(X, Y):
    n=len(X)
    Sx=sum(X)   # Sun of all x
    Sy=sum(Y)   # Sun of all y

    x_mean=sum(X)/n    # Mean x
    y_mean=sum(Y)/n    # Mean y

    Sxx=0
    Sxy=0
    Syy=0
    for i in range(len(X)):
        Sxx += (X[i] - x_mean)**2
        Sxy += (X[i] - x_mean) * (Y[i] - y_mean)
        Syy += (Y[i] - y_mean)**2
    return n, x_mean, y_mean, Sx, Sy, Sxx, Syy, Sxy



# Function to calculate Pearson Coefficient

def Pearson_coeff(X, Y):
    S=find_stats(X,Y)
    r2 = S[7]**2 / (S[5] * S[6])
    r = r2**(0.5)

    return r


########################################################################################################################


# solve for m and c

def Line_fit(X, Y):
    n = len(X) # or len(Y)
    xbar = sum(X)/n
    ybar = sum(Y)/n

    # Calculating numerator and denominator
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    # calculation of slope and intercept
    m = numer / denum
    c = ybar - m * xbar
    return c, m



# Plotting the graph

def plot_graph_linear(X, Y, c, m):
    plt.figure(figsize=(7,5))
    # plot points and fit line
    plt.scatter(X, Y, s=50, color='blue')
    yfit = [c + m * xi for xi in X]
    plt.plot(X, yfit, 'r-', label="Best fit line")


########################################################################################################################
    

# Ploynomial fit with given degree

# def polynomial_fitting(X,Y, order):
#     X1=copy.deepcopy(X)
#     Y1=copy.deepcopy(Y)
#     order+=1
    
#     # Finding the coefficient matrix - refer notes
#     A=[[0 for j in range(order)] for i in range(order)]
#     vector=[0 for i in range(order)]

#     for i in range(order):
#         for j in range(order):
#             for k in range(len(X)):
#                 A[i][j] += X[k]**(i+j)

#     Det=determinant(A,order)
#     print("Determinant is = "+ str(Det))
#     if Det==0:
#         print("Determinant is zero. Inverse does not exist")
#     print("Determinant is not zero. Inverse exists.\n")
#     # Finding the coefficient vector - refer notes
#     for i in range(order):
#         for k in range(len(X)):
#             vector[i] += X[k]**i * Y[k]

#     # Solution finding using LU decomposition using Doolittle's condition L[i][i]=1
#     # partial pivoting to avoid division by zero at pivot place
#     A, vector = partial_pivot_LU(A, vector, order)
#     A = LU_doolittle(A,order)
    
#     # Finding coefficient vector
#     solution = for_back_subs_doolittle(A,order,vector)

#     return solution[0:order]

        

# Plotting the graph

def plot_graph_poly(X, Y, sol, order):
    yfit=[0 for i in range(len(X))]
    # finding yfit
    for k in range(len(X)):
        for l in range(order):
            yfit[k]+=sol[l]*X[k]**l
    
    # plotting X and y_fit
    plt.plot(X, yfit, 'r-', label="Curve fit with polynomial of degree = "+ str(order-1))


########################################################################################################################
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
########################################################################################################################
""" END OF LIBRARY """
########################################################################################################################
