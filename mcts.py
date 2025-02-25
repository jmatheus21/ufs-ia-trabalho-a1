import math
import random
import numpy as np

class Node:
    def __init__(self, state, parent=None, done=False):
        self.state = state
        self.parent = parent
        self.done = done
        self.children = []
        self.visits = 0
        self.wins = 0

class MCTS:
    def __init__(self, env, num_simulations=100):
        self.env = env
        self.num_simulations = num_simulations

    # foi utilizado o valor de 2 na constante de exploração, para fazer o MCTS não ficar preso em estados subótimos
    def uct_select(self, node, exploration_constant=2):
        best_score = -math.inf
        best_children = []
        for action, child_node in node.children:
            if child_node.visits == 0:
                score = math.inf
            else:
                exploitation = child_node.wins / child_node.visits
                exploration = exploration_constant * math.sqrt(math.log(node.visits) / child_node.visits)
                score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_children = [(action, child_node)]
            elif score == best_score:
                best_children.append((action, child_node))
        return random.choice(best_children)

    def expand(self, node):
        for action in range(self.env.action_space.n):
            actions = np.array([action] * self.env.num_envs)
            next_states, _, dones, _ = self.env.step(actions)
            
            for i in range(self.env.num_envs):
                if not dones[i]:
                    child_node = Node(next_states[i], parent=node, done=dones[i])
                    node.children.append((action, child_node))

    def simulate(self, node):
        state = node.state
        trajectory = []
        done = False
        
        while not done:
            # lógica definida para fazer com que o agente se movimente e ataque mais
            if not self.has_enemies_in_front(state):
                action = random.choice([2, 3])
            else:
                action = random.choices([1, 2, 3, 0], weights=[50, 20, 20, 10])[0]
            
            actions = np.array([action] * self.env.num_envs)
            next_states, rewards, dones, _ = self.env.step(actions)
            trajectory.append((state, action, rewards[0], next_states[0], dones[0]))
            state = next_states[0]
            done = dones[0]
        
        return trajectory

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

    def generate_trajectories(self, state, num_trajectories=10):
        trajectories = []
        for i in range(num_trajectories):
            root = Node(state)
            for j in range(self.num_simulations):
                print(f"simulação n° {j+1} da trajetória {i}")
                node = root
                # Seleção
                while node.children and not node.done:
                    action, node = self.uct_select(node)

                # Expansão
                if not node.children and not node.done:
                    self.expand(node)

                # Simulação
                if node.children:
                    action, node = random.choice(node.children)
                trajectory = self.simulate(node)
                trajectories.extend(trajectory)

                # Retropropagação
                result = sum([reward for _, _, reward, _, _ in trajectory])
                self.backpropagate(node, result)
        return trajectories

    # método criado para detectar inimigos na frente do jogador
    def has_enemies_in_front(self, state):
        current_frame = state[:, :, -1]
        enemy_intensity = 134 

        enemy_region = current_frame[:-20, :]

        enemy_pixels = enemy_region == enemy_intensity
        return np.any(enemy_pixels)