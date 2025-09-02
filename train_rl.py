# train_rl.py
from core.rl.environment import MarketEnvironment
from core.rl.agent import DQNAgent
from core.data import HistoricalDataHandler
from core.portfolio import Portfolio
from config import Config, BacktestConfig

def main():
    data_handler = HistoricalDataHandler(
        symbols=[Config.UNDERLYING_SYMBOL],
        start_date=BacktestConfig.START_DATE,
        end_date=BacktestConfig.END_DATE,
        timeframe=Config.HISTORICAL_DATA_TIMEFRAME
    )
    portfolio = Portfolio(initial_capital=BacktestConfig.INITIAL_CAPITAL)
    env = MarketEnvironment(data_handler, portfolio)
    
    state_size = env.data.shape[1] - 5 # Exclude OHLCV
    agent = DQNAgent(state_size=state_size, action_size=3)
    
    episodes = 10
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        for time in range(env.n_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
                
    # Save the trained model
    torch.save(agent.model.state_dict(), 'rl_trading_bot_dqn.pt')

if __name__ == "__main__":
    main()