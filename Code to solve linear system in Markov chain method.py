# Code to solve linear system in the Markov chain method
import numpy as np
def state_to_index(t_A, t_B, P):
    # Map Markov chain states to index
    # t_A,t_B have 5 choices, P has 2 choices, leading to 50 states
    """
    We order them as:
    (t_A=1,t_B=1,A), (t_A=1,t_B=1,B),
    (t_A=1,t_B=2,A), (t_A=1,t_B=2,B),
    ...
    (t_A=5,t_B=5,A), (t_A=5,t_B=5,B)
    """
    # P: 'A' -> 0, 'B' -> 1
    return ((t_A - 1)*5 + (t_B - 1))*2 + (0 if P=='A' else 1)

def build_PA_system(p_a, p_b, q_a, q_b):
    # Build the linear system A_p * P_A = b_p for P_A(t_A,t_B,P).
    A_p = np.zeros((50,50))
    b_p = np.zeros(50)
    
    for tA in range(1,6):
        for tB in range(1,6):
            for P in ['A','B']:
                i = state_to_index(tA,tB,P)
                if P=='A':
                    # Consider transition equation for P_A(tA,tB,A):
                    # P_A(tA,tB,A)= p_a q_b P_A(tA,tB,B) + p_a(1-q_b)P_A(tA,tB-1,B)
                    #   + (1-p_a)q_b P_A(tA,tB,B) + (1-p_a)(1-q_b)P_A(tA-1,tB,B)
                    # Combine terms and move all unknowns to LHS:
                    
                    A_p[i,i] = 1.0 # Coeff for P_A(tA,tB,A)
                    
                    # P_A(tA,tB,B)
                    j = state_to_index(tA,tB,'B')
                    # Coeff for P_A(tA,tB,B) on RHS: p_a q_b + (1-p_a)q_b = q_b
                    A_p[i,j] -= (p_a*q_b + (1-p_a)*q_b)
                    
                    # P_A(tA,tB-1,B) if tB-1>=1:
                    if tB-1 >=1:
                        j = state_to_index(tA,tB-1,'B')
                        A_p[i,j] -= p_a*(1-q_b)
                    else:
                        # tB-1=0 => P_A(tA,0,B)=1 terminal (A wins)
                        b_p[i] += p_a*(1-q_b)*1.0
                    
                    # P_A(tA-1,tB,B) if tA-1>=1:
                    if tA-1>=1:
                        j = state_to_index(tA-1,tB,'B')
                        A_p[i,j] -= (1-p_a)*(1-q_b)
                    else:
                        # tA-1=0 => P_A(0,tB,B)=0 terminal (B wins)
                        b_p[i] += (1-p_a)*(1-q_b)*0.0
                    
                else:
                    # P='B'
                    # Doing same things as above
                    
                    A_p[i,i] = 1.0 # For P_A(tA,tB,B)
                    
                    # P_A(tA,tB,A)
                    j = state_to_index(tA,tB,'A')
                    A_p[i,j] -= (p_b*q_a + (1-p_b)*q_a) # = q_a
                    
                    # P_A(tA-1,tB,A) if tA-1>=1
                    if tA-1>=1:
                        j = state_to_index(tA-1,tB,'A')
                        A_p[i,j] -= p_b*(1-q_a)
                    else:
                        # tA-1=0 => P_A(0,tB,A)=0
                        b_p[i] += p_b*(1-q_a)*0.0
                        
                    # P_A(tA,tB-1,A) if tB-1>=1
                    if tB-1>=1:
                        j = state_to_index(tA,tB-1,'A')
                        A_p[i,j] -= (1-p_b)*(1-q_a)
                    else:
                        # tB-1=0 => P_A(tA,0,A)=1
                        b_p[i] += (1-p_b)*(1-q_a)*1.0
                        
    return A_p, b_p

def build_W_system(P_A_vec, p_a, p_b, q_a, q_b, winner='A'):
    # Build the linear system of W_A,W_B
    A_w = np.zeros((50,50))
    b_w = np.zeros(50)
    
    def P_A_func(tA,tB,P):
        # Terminal conditions for P_A:
        if tB==0: return 1.0
        if tA==0: return 0.0
        return P_A_vec[state_to_index(tA,tB,P)]
    
    for tA in range(1,6):
        for tB in range(1,6):
            for P in ['A','B']:
                i = state_to_index(tA,tB,P)
                A_w[i,i] = 1.0
                pa_val = P_A_func(tA,tB,P) # Probability that A eventually wins
                pb_val = 1 - pa_val # Probability that B eventually wins
            
                if winner=='A':
                    # Transition equation: W_A(tA,tB,P) = pa_val + ...
                    rhs = pa_val
                else:
                    # Some for W_B
                    rhs = pb_val
                
                # Doing same things as same as P_A
                if P=='A':
                    if True: # always non-terminal if tA,tB>0
                        j = state_to_index(tA,tB,'B')
                        A_w[i,j] -= (p_a*q_b + (1-p_a)*q_b) # q_b
                        
                    # (tA,tB-1,B) if tB-1>=1
                    if tB-1>=1:
                        j = state_to_index(tA,tB-1,'B')
                        A_w[i,j] -= p_a*(1-q_b)
                    # if tB-1=0 => terminal => W_winner(tA,0,B)=0 no addition needed
                        
                    # (tA-1,tB,B) if tA-1>=1
                    if tA-1>=1:
                        j = state_to_index(tA-1,tB,'B')
                        A_w[i,j] -= (1-p_a)*(1-q_b)
                    # if tA-1=0 => terminal => W_winner(0,tB,B)=0 no addition 
                else:
                    j = state_to_index(tA,tB,'A')
                    A_w[i,j] -= (p_b*q_a + (1-p_b)*q_a) # q_a
                    if tA-1>=1:
                        j = state_to_index(tA-1,tB,'A')
                        A_w[i,j] -= p_b*(1-q_a)
                    # tA-1=0 => terminal W_winner(0,tB,A)=0
                    if tB-1>=1:
                        j = state_to_index(tA,tB-1,'A')
                        A_w[i,j] -= (1-p_b)*(1-q_a)
                    # tB-1=0 => terminal W_winner(tA,0,A)=0
                b_w[i] = rhs
    return A_w, b_w

# Set given probabilities:
p_a, p_b, q_a, q_b = 0.7,0.4,0.55,0.37 
# Solve for P_A
A_p, b_p = build_PA_system(p_a, p_b, q_a, q_b)
P_As = np.linalg.solve(A_p, b_p)
def P_A_func(tA,tB,P):
    if tB==0: return 1.0
    if tA==0: return 0.0
    return P_As[state_to_index(tA,tB,P)]

#Solve for W_A
A_wA, b_wA = build_W_system(P_As, p_a, p_b, q_a, q_b, winner='A')
W_As = np.linalg.solve(A_wA, b_wA)  
def W_A_func(tA,tB,P):
    if tA==0 or tB==0:
        return 0.0
    return W_As[state_to_index(tA,tB,P)]
    
# Solve for W_B
A_wB, b_wB = build_W_system(P_As, p_a, p_b, q_a, q_b, winner='B')
W_Bs = np.linalg.solve(A_wB,b_wB)
def W_B_func(tA,tB,P):
    if tA==0 or tB==0:
        return 0.0
    return W_Bs[state_to_index(tA,tB,P)]

# Calculate expectations
P_A = P_A_func(5,5,'A')
W_A = W_A_func(5,5,'A')
E_T_given_A_wins = W_A / P_A if P_A>0 else 0.0
W_B = W_B_func(5,5,'A')
P_B = 1.0 - P_A
E_T_given_B_wins = W_B / P_B if P_B>0 else 0.0
    
# Print results:
print("E[T|A wins]:", E_T_given_A_wins)
print("E[T|B wins]:", E_T_given_B_wins)