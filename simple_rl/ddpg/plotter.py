from simple_rl.common.logger import Logger

import matplotlib.pyplot as plt



def plot(logger:Logger, figsize:tuple=(13,6), linewidth:float=0.4) :
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    
    red_line = dict(
        color='tab:red',
        linewidth=linewidth
    )
    blue_line = dict(
        color='tab:blue',
        linewidth=linewidth
    )
    
    # 1th row
    axs[0,0].plot(*logger.get('episode/r'), **red_line)
    axs[0,0].set_xlabel('timestep')
    axs[0,0].set_ylabel('return')

    axs[0,1].plot(*logger.get('episode/l'), **red_line)
    axs[0,1].set_xlabel('timestep')
    axs[0,1].set_ylabel('length')

    axs[0,2].plot(*logger.get('episode/t'), **red_line)
    axs[0,2].set_xlabel('timestep')
    axs[0,2].set_ylabel('time')

    # 2th row
    axs[1,0].plot(*logger.get('train/actor_lr'), **red_line, label='actor lr')
    axs[1,0].plot(*logger.get('train/critic_lr'), **blue_line, label='critic lr')
    axs[1,0].set_xlabel('timestep')
    axs[1,0].set_ylabel('learning rate')
    axs[1,0].legend()

    axs[1,1].plot(*logger.get('train/actor_loss'), **red_line)
    axs[1,1].set_xlabel('timestep')
    axs[1,1].set_ylabel('actor loss')

    axs[1,2].plot(*logger.get('train/critic_loss'), **red_line)
    axs[1,2].set_xlabel('timestep')
    axs[1,2].set_ylabel('critic loss')

    plt.tight_layout()
    plt.show()

    return