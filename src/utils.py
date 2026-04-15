def log(message):
    print(f"[LOG] {message}")

def save_model(agent, filename):
    save_dir = filename.rsplit('.', 1)[0]
    agent.save_model(save_dir)
    log(f"Model saved to {save_dir}")

def load_model(filename):
    from src.environment import DroneEnvironment
    env = DroneEnvironment()
    agent = DroneAgent(env)
    agent.load_model(filename)
    log(f"Model loaded from {filename}")
    return agent

def plot_training_progress(rewards, filename='training_progress.png'):
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Episodes')
    plt.ylabel('Total Rewards')
    plt.savefig(filename)
    log(f"Training progress saved to {filename}")