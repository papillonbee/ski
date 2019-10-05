from ski import ski
from scipy.stats import poisson
from numpy import random
import matplotlib.pyplot as plt

sales = None
wholesale = None
retail = None
r = None
alpha = None
N = None
M = None
lam = None

poi = None

opt_reward = {}
opt_action = {}
reward = {}

est_opt_reward = {}
est_opt_action = {}
est_reward = {}
visit = {}

x = []
y = []
total_episodes = 0

def reset():
    global opt_reward,opt_action,reward,x,y,total_episodes,est_opt_reward,est_opt_action,est_reward,visit
    opt_reward = {}
    opt_action = {}
    reward = {}
    x = []
    y = []
    total_episodes = 0
    est_opt_reward = {}
    est_opt_action = {}
    est_reward = {}
    visit = {}
    
def set_poi():
    if lam != None:
        global poi
        poi = poisson(lam)

def f(t,inventory):
    state = (t,inventory)
    if state in opt_reward:
      return opt_reward[state]
    if t == N:
      opt_reward[state] = sales*inventory
      return opt_reward[state]
    R = 0
    actions = {}
    for buy in range(M-inventory+1):
      expected_demand = 0
      expected_future_profit = 0
      for demand in range(inventory+buy+1):
        if demand == inventory+buy:
          expected_demand += demand*(1-poi.cdf(demand-1))
          expected_future_profit += f(t+1,inventory+buy-demand)*(1-poi.cdf(demand-1))
        else:
          expected_demand += demand*poi.pmf(demand)
          expected_future_profit += f(t+1,inventory+buy-demand)*poi.pmf(demand)
      expected_demand *= alpha*retail
      cost = wholesale*buy+alpha*r*wholesale*(inventory+buy)
      profit = expected_demand-cost
      acc_profit = profit+alpha*expected_future_profit
      actions[buy] = acc_profit
      R = max(R,acc_profit)
    opt_reward[state] = R
    opt_action[state] = max(actions,key=actions.get)
    reward[state] = actions
    return opt_reward[state]

def solve_with_mdp(start_state):
    t = start_state[0]
    inventory = start_state[1]
    f(t,inventory)
    return {'opt_reward': opt_reward,'opt_action': opt_action,'reward': reward}

def epsilon(episode,epsilon_greedy):
    if epsilon_greedy == 'decayed':
        e = 1/(1+episode/1000)
        return e
    return epsilon_greedy

def buy_policy(episode,cap,state,epsilon_greedy):
    rand = random.rand()
    if state not in est_reward or rand < epsilon(episode,epsilon_greedy):
        return random.randint(0,cap+1)
    return max(est_reward[state],key=est_reward[state].get)

def plot():
    plt.plot(x,y)
    plt.show()

def solve_with_mc(start_state,episodes,keep_result,epsilon_greedy):
    global x,y,total_episodes,est_opt_reward,est_opt_action,est_reward,visit
    if not keep_result:
        est_opt_reward = {}
        est_opt_action = {}
        est_reward = {}
        visit = {}
        x = []
        y = []
        total_episodes = 0
    for episode in range(total_episodes,total_episodes+episodes):
        inventory = start_state[1]
        R = 0
        demands = []
        buys = []
        inventories = []
        for t in range(start_state[0],N+1):
            state = (t,inventory)
            if state not in visit:
                visit[state] = {}
            demand = poi.rvs()
            buy = buy_policy(episode,M-inventory,state,epsilon_greedy)
            if buy not in visit[state]:
                visit[state][buy] = 1
            else:
                visit[state][buy] += 1
            demands.append(demand)
            buys.append(buy)
            inventories.append(inventory)
            inventory = max(inventory+buy-demand,0)
        for t in range(N,start_state[0]-1,-1):
            if t == N:
                state = (t,inventories[t-1])
                if state not in est_opt_reward:
                    est_opt_reward[state] = sales*inventories[t-1]
                R += est_opt_reward[state]
            else:
                R *= alpha
                R += alpha*retail*min(demands[t-1],inventories[t-1]+buys[t-1])-wholesale*buys[t-1]-alpha*r*wholesale*(inventories[t-1]+buys[t-1])
                state = (t,inventories[t-1])
                action = buys[t-1]
                if state not in est_reward:
                    est_reward[state] = {}
                if action not in est_reward[state]:
                    est_reward[state][action] = 0
                n = visit[state][action]
                est_reward[state][action] = ((n-1)*est_reward[state][action]+R)/(n)
        y.append(max(est_reward[start_state].values()))
    total_episodes += episodes
    x = [episode for episode in range(total_episodes)]
    for state in est_reward:
        est_opt_reward[state] = max(est_reward[state].values())
        est_opt_action[state] = max(est_reward[state],key=est_reward[state].get)
    return {'est_opt_reward': est_opt_reward,'est_opt_action': est_opt_action,'est_reward': est_reward,'visit': visit,'plot': plot}