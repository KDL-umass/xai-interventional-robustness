mkdir -p storage/models/Amidar/vqn
mkdir -p storage/models/Amidar/vsarsa
mkdir -p storage/models/Amidar/c51
mkdir -p storage/models/Amidar/a2c
mkdir -p storage/models/Amidar/ppo
mkdir -p storage/models/Amidar/dqn
mkdir -p storage/models/Amidar/ddqn
mkdir -p storage/models/Amidar/rainbow

mkdir -p storage/models/Breakout/vqn
mkdir -p storage/models/Breakout/vsarsa
mkdir -p storage/models/Breakout/c51
mkdir -p storage/models/Breakout/a2c
mkdir -p storage/models/Breakout/ppo
mkdir -p storage/models/Breakout/dqn
mkdir -p storage/models/Breakout/ddqn
mkdir -p storage/models/Breakout/rainbow

mkdir -p storage/models/SpaceInvaders/vqn
mkdir -p storage/models/SpaceInvaders/vsarsa
mkdir -p storage/models/SpaceInvaders/c51
mkdir -p storage/models/SpaceInvaders/a2c
mkdir -p storage/models/SpaceInvaders/ppo
mkdir -p storage/models/SpaceInvaders/dqn
mkdir -p storage/models/SpaceInvaders/ddqn
mkdir -p storage/models/SpaceInvaders/rainbow

scp -r gypsum-remote:'/mnt/nfs/work1/jensen/kavery/SpaceInvaders/vqn/*'    storage/models/SpaceInvaders/vqn
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/kavery/Amidar/vqn/*'   storage/models/Amidar/vqn
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/kavery/Breakout/vqn/*' storage/models/Breakout/vqn
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/jnkenney/SpaceInvaders/vsarsa/*'   storage/models/SpaceInvaders/vsarsa
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/jnkenney/Amidar/vsarsa/*'  storage/models/Amidar/vsarsa
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/jnkenney/Breakout/vsarsa/*'    storage/models/Breakout/vsarsa
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/pboddavarama/xai-agents/SpaceInvaders/ppo*'   storage/models/SpaceInvaders/ppo
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/pboddavarama/xai-agents/Amidar/ppo*'  storage/models/Amidar/ppo
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/pboddavarama/xai-agents/Breakout/ppo*'    storage/models/Breakout/ppo
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/ecai/SpaceInvaders/dqn/*'  storage/models/SpaceInvaders/dqn
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/ecai/Amidar/dqn/*' storage/models/Amidar/dqn
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/ecai/Breakout/dqn/*'   storage/models/Breakout/dqn
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/ecai/SpaceInvaders/a2c/*'  storage/models/SpaceInvaders/a2c
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/ecai/Amidar/a2c/*' storage/models/Amidar/a2c
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/ecai/Breakout/a2c/*'   storage/models/Breakout/a2c
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/pboddavarama/xai-agents/SpaceInvaders/rainbow*'   storage/models/SpaceInvaders/rainbow
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/pboddavarama/xai-agents/Amidar/rainbow*'  storage/models/Amidar/rainbow
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/pboddavarama/xai-agents/Breakout/rainbow*'    storage/models/Breakout/rainbow
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/ecai/SpaceInvaders/c51/*'  storage/models/SpaceInvaders/c51
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/ecai/Amidar/c51/*' storage/models/Amidar/c51
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/ecai/Breakout/c51/*'   storage/models/Breakout/c51
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/jnkenney/SpaceInvaders/ddqn/*' storage/models/SpaceInvaders/ddqn
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/pboddavarama/xai-agents/Amidar/ddqn*'  storage/models/Amidar/ddqn
scp -r gypsum-remote:'/mnt/nfs/work1/jensen/jnkenney/Breakout/ddqn/*'  storage/models/Breakout/ddqn