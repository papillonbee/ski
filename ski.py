from ski import private

def set_params(sales=None,wholesale=None,retail=None,r=None,alpha=None,N=None,M=None,lam=None):
    private.sales = sales
    private.wholesale = wholesale
    private.retail = retail
    private.r = r
    private.alpha = alpha
    private.N = N
    private.M = M
    private.lam = lam
    private.set_poi()
    private.reset()

def solve(method='mdp',start_state=(1,0),episodes=10000,keep_result=False,epsilon_greedy=.05):
    if method == 'mdp':
        return private.solve_with_mdp(start_state)
    if method == 'mc':
        return private.solve_with_mc(start_state,episodes,keep_result,epsilon_greedy)