import pickle
import neat

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            "./neat-config")
with open("./winner.pkl", "rb") as f:
    genome = pickle.load(f)

net = neat.nn.FeedForwardNetwork.create(genome, config)
def neat_weigh(occ):
    out = net.activate((occ.center_angle, occ.center_point[0], occ.center_point[1], occ.distance, occ.relative_velocity[0], occ.relative_velocity[1], occ.relative_speed))
    return out[0]