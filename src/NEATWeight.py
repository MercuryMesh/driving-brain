from time import time
from airsim.client import CarClient
import neat
from MainClass import Main
from daq.VisionDelegate import VisionDelegate
from drivers.CollisionWatchdog import CollisionStrategy, CollisionWatchdog
from drivers.DrivingArbiter import DrivingArbiter
from managers.AngularOccupancy import DISCRETIZATION_AMOUNT, AngularOccupancy
import pickle


IDLE_THRESHOLD = 5
IDLE_SPEED = 1
GOAL_Y = -48
class NeatTrainer:
    def __init__(self):
        self.carClient = CarClient()
        self.createAll()
    def createAll(self):
        self.drivingArbiter = DrivingArbiter(self.carClient)
        self.visionDelegate = VisionDelegate('./models/model3.tflite', 16, 0.7)
        self.angularOccupancy = AngularOccupancy(visionDelegate=self.visionDelegate)
        self.collisionWatchdog = CollisionWatchdog(self.drivingArbiter, self.angularOccupancy)
        self.mainLooper = Main(self.carClient, self.collisionWatchdog, self.drivingArbiter, self.angularOccupancy, self.visionDelegate)

    def initData(self):
        self._startPos = self.carClient.simGetVehiclePose().position

    def eval_genomes(self, genomes, config):
        for id, genome in genomes:
            print(f"Running genome {id}")
            self.carClient.reset()
            self.collisionWatchdog._collisionStrategy = CollisionStrategy.none
            self.drivingArbiter.giveUpSpeedControl(self.collisionWatchdog)
            self.drivingArbiter.giveUpSteeringControl(self.collisionWatchdog)
            self.initData()
            genome.fitness = 100.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            last_loop_time = time()
            collision_avoidance_time = 0.0
            idle_time = 0.0
            start_time = time()
            loop_iterations = 0
            while True:
                loop_iterations += 1
                collision = self.carClient.simGetCollisionInfo()
                state = self.carClient.getCarState()
                delta_time = time() - last_loop_time
                if state.speed <= IDLE_SPEED:
                    idle_time += delta_time
                else:
                    idle_time = 0
                
                if idle_time >= IDLE_THRESHOLD:
                    print("Idle exiting")
                    break

                if collision.has_collided:
                    genome.fitness /= 5
                    print("Collision exiting")
                    break

                if self.collisionWatchdog._collisionStrategy != CollisionStrategy.none:
                    collision_avoidance_time += delta_time
                
                for key in self.angularOccupancy.occupant_reference:
                    occ = self.angularOccupancy.occupant_reference[key]
                    out = net.activate((occ.center_angle, occ.center_point[0], occ.center_point[1], occ.distance, occ.relative_velocity[0], occ.relative_velocity[1], occ.relative_speed))
                    occ.set_weight(out[0])
                last_loop_time = time()
                self.mainLooper.main()
            print(f"Loop execution time {(time() - start_time) / loop_iterations}s")
            endPos = self.carClient.simGetVehiclePose().position
            off_goal = GOAL_Y - endPos.y_val
            genome.fitness -= (off_goal) ** 2 + (collision_avoidance_time)

                
    def run(self, config_file):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)
        # p = neat.Population(config)
        p = neat.Checkpointer.restore_checkpoint('./neat-checkpoint-15')
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        w = p.run(self.eval_genomes, 100)

        print('\nBest genome:\n{!s}'.format(w))
        
        with open("winner.pkl", "wb") as f:
            pickle.dump(w, f)

if __name__ == "__main__":
    nw = NeatTrainer()
    nw.run("./neat-config")
    