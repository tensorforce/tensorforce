if [ -z ${2+x} ]; then

    echo "=== Benchmarking $1 ==="

    echo "OpenAI Gym: classic control"
    # echo "> Acrobot-v1"
    # python run.py benchmarks/configs/$1.json gym -l Acrobot-v1 -e 300 -r 10 -p benchmarks/gym-acrobot/$1
    echo "> CartPole-v1"
    python run.py benchmarks/configs/$1.json gym -l CartPole-v1 -e 100 -r 10 -p benchmarks/gym-cartpole/$1
    # echo "> MountainCar-v0"
    # python run.py benchmarks/configs/$1.json gym -l MountainCar-v0 -e 300 -r 10 -p benchmarks/gym-mountaincar/$1
    # echo "> MountainCarContinuous-v0"
    # python run.py benchmarks/configs/$1.json gym -l MountainCarContinuous-v0 -e 300 -r 10 -p benchmarks/gym-mountaincar-continuous/$1
    # echo "> Pendulum-v0"
    # python run.py benchmarks/configs/$1.json gym -l Pendulum-v0 -e 300 -r 10 -p benchmarks/gym-pendulum/$1

    # echo "OpenAI Gym: Box2D"
    # echo "> LunarLander-v2"
    # python run.py benchmarks/configs/$1.json gym -l LunarLander-v2 -e 300 -r 10 -p benchmarks/gym-lunarlander/$1

else

    python run.py benchmarks/configs/$1.json benchmarks/configs/$2.json -e 300 -r 10 -p benchmarks/$2/$1

fi
