# Code to simulate the game
import random
def simulate_game(p_a, q_a, p_b, q_b, num_tokens=5):
    tokens_a = num_tokens
    tokens_b = num_tokens
    rounds = 0
    drawer = 'A'  # Start with Player A
    responder = 'B'
    while tokens_a > 0 and tokens_b > 0:
        rounds += 1
        if drawer == 'A':
            # Player A is the drawer, B is the responder
            is_truth = random.random() < p_a
            accept = random.random() < q_b
            if accept:
                pass  # No tokens lost
            else:
                if is_truth:
                    # B loses a token
                    tokens_b -= 1
                else:
                    # A loses a token
                    tokens_a -= 1
        else:
            # Player B is the drawer, A is the responder
            is_truth = random.random() < p_b
            accept = random.random() < q_a
            if accept:
                pass  # No tokens lost
            else:
                if is_truth:
                    # A loses a token
                    tokens_a -= 1
                else:
                    # B loses a token
                    tokens_b -= 1
        # Switch roles for the next turn
        drawer, responder = responder, drawer
    winner = 'A' if tokens_b <= 0 else 'B'
    return rounds, winner

def simulate_multiple_games(p_a_truth, q_a_accept, p_b_truth, q_b_accept, num_simulations=1000):
    total_rounds_a = 0
    total_rounds_b = 0
    wins_a = 0
    wins_b = 0
    for _ in range(num_simulations):
        rounds, winner = simulate_game(p_a_truth, q_a_accept, p_b_truth, q_b_accept)
        if winner == 'A':
            total_rounds_a += rounds
            wins_a += 1
        else:
            total_rounds_b += rounds
            wins_b += 1    
    expected_rounds_a = total_rounds_a / wins_a if wins_a > 0 else float('inf')
    expected_rounds_b = total_rounds_b / wins_b if wins_b > 0 else float('inf')
    print(f"Player A expected rounds to win: {expected_rounds_a:.2f}")
    print(f"Player B expected rounds to win: {expected_rounds_b:.2f}")

p_a = 0.7  # Probability Player A tells the truth when they are the drawer
q_a = 0.55  # Probability Player A accepts the claim when they are the responder
p_b = 0.4 # Probability Player B tells the truth when they are the drawer
q_b = 0.37 # Probability Player B accepts the claim when they are the responder

simulate_multiple_games(p_a, q_a, p_b, q_b)