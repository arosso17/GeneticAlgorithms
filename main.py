from Car import *

win = pg.display.set_mode((800, 600))
clock = pg.time.Clock()

def selection(cars):  # returns the top two performing cars of a generation
    sortedlist = sorted(cars, key=lambda x: x.score, reverse=True)
    return sortedlist[0], sortedlist[1]

def crossover(c1, c2, checkpoints):
    c1 = Car([100, 500], c1.nn.weights[:], c1.nn.biases[:], checkpoints[:])
    c2 = Car([100, 500], c2.nn.weights[:], c2.nn.biases[:], checkpoints[:])
    c3w = []
    c4w = []
    for par1, par2 in zip(c1.nn.weights, c2.nn.weights):
        crosspoint = int(np.random.rand() * len(c1.nn.weights))
        t3 = []
        t4 = []
        for p in par1[0:crosspoint]:
            t3.append(p)
        for p in par2[crosspoint:]:
            t3.append(p)
        for p in par2[0:crosspoint]:
            t4.append(p)
        for p in par2[crosspoint:]:
            t4.append(p)
        c3w.append(t3)
        c4w.append(t4)
    c3 = Car([100, 500], c3w, c1.nn.biases[:], checkpoints[:])
    c4 = Car([100, 500], c4w, c1.nn.biases[:], checkpoints[:])
    return [c1, c2, c3, c4]

def main():
    run = True

    # roads = [Road([50, 50], [100, 500], 2), Road([50, 50], [700, 100], 1),
    #          Road([650, 50], [100, 500], 4), Road([50, 450], [700, 100], 3)]    # basic right turn loop

    roads = [Road([50, 50], [100, 500], 2), Road([50, 50], [250, 100], 1), Road([250, 50], [100, 350], 4),
             Road([250, 300], [300, 100], 1), Road([450, 50], [100, 300], 2), Road([450, 50], [200, 100], 1),
             Road([650, 50], [100, 500], 4), Road([50, 450], [700, 100], 3)]      # training course

    # roads = [Road([50, 300], [100, 250], 2), Road([50, 300], [500, 100], 1), Road([500, 150], [100, 250], 2),
    #          Road([50, 150], [500, 100], 3), Road([50, 0], [100, 250], 2), Road([50, 0], [700, 100], 1),
    #          Road([650, 0], [100, 550], 4), Road([50, 450], [700, 100], 3)]     # testing course

    # roads = [Road([50, 450], [700, 100], 1), Road([650, 50], [100, 500], 2), Road([450, 50], [200, 100], 3),
    #          Road([450, 50], [100, 300], 4), Road([250, 300], [300, 100], 3), Road([250, 50], [100, 350], 2),
    #          Road([50, 50], [250, 100], 3), Road([50, 50], [100, 500], 4)]     # inverse of training course

    checkpoints = []
    for road in roads:
        checkpoints += road.get_checkpoints()
    print(checkpoints)
    print(len(checkpoints))
    num = 100                                                                  # population size
    cars = [Car([100, 500], None, None, checkpoints[:]) for _ in range(num)]  # random initialization
    lines = False
    draw = True
    runs = 0
    old_best = cars[0]
    while run:
        win.fill("green")
        for road in roads:
            road.draw(win)
        if old_best:
            old_best.draw(win, False, True)
        num_crashed = 0
        for car in cars:
            if not car.crashed:
                for p in car.points:
                    on_road = False
                    for road in roads:
                        if road.rect.collidepoint(p[0], p[1]):
                            on_road = True
                    if not on_road:
                        car.crashed = True
                        # best_car, second_best = selection(cars)
                    # print("number crashed:", num_crashed, "out of:", len(cars))
                car.update(win, lines, draw, roads)
                if car.time > 1000000:
                    car.crashed = True
                if not car.cp:
                    car.cp = checkpoints[:]
            if car.crashed:
                num_crashed += 1
            if draw:
                car.draw(win, lines, draw)
        best_car, second_best = selection(cars)
        old_best.draw(win, lines, True, True)
        best_car.draw(win, lines, True, True)
        if num_crashed == len(cars) or (best_car.score > len(checkpoints) * 250):
            runs += 1
            # best_car, second_best = selection(cars)

            print("Generation: ", runs)
            # print("high score: ", best_car.score, best_car)
            # print("2nd high score: ", second_best.score, second_best)

            old_best = best_car
            if best_car.score > (len(checkpoints) + 15) * 100:
                print()
                print()
                print("WIN")
                # print(best_car.nn.biases)
                # print(best_car.nn.weights)
                print()
                print()
                ncars = [Car([100, 500], None, None, checkpoints[:]) for _ in range(num)]
                cars = ncars[:]
                runs = 0
                old_best = cars[0]
            else:
                ncars = crossover(best_car, second_best, checkpoints)  # with crossover
                # ncars = [best_car, second_best]                      # without crossover
                temp = []
                for car in ncars:
                    w, b = car.nn.variant()
                    w = w[:]
                    b = b[:]
                    temp.append(Car([100, 500], w[:], b[:], checkpoints[:]))
                for t in temp:
                    ncars.append(t)
                # ncars.append(Car([100, 500], None, None, checkpoints[:]))  # can add a couple random cars each gen
                # ncars.append(Car([100, 500], None, None, checkpoints[:]))  # to increase genetic diversity
                for _ in range(len(cars) - len(ncars)):
                    w, b = best_car.nn.mutate()
                    w = w[:]
                    b = b[:]
                    ncars.append(Car([100, 500], w[:], b[:], checkpoints[:]))
                cars = ncars[:]

        pg.display.flip()
        clock.tick()
        pg.display.set_caption(str(round(clock.get_fps())))
        for event in pg.event.get():
            if event.type == pg.KEYUP:
                if event.key == pg.K_l:  # show the "vision" lines of the cars
                    lines = not lines
                if event.key == pg.K_d:  # toggles drawing all cars or just the current best
                    draw = not draw
                if event.key == pg.K_n:  # manually moves on to the next generation
                    best_car, second_best = selection(cars)
                    runs += 1
                    print("run number: ", runs)
                    print("high score: ", best_car.score)
                    old_best = best_car
                    ncars = [Car([100, 500], best_car.nn.weights, best_car.nn.biases, checkpoints[:])]
                    for _ in range(len(cars) - 1):
                        w, b = best_car.nn.evolve()
                        ncars.append(Car([100, 500], w, b, checkpoints[:]))
                    cars = ncars
                if event.key == pg.K_p:  # prints the weights and biases of the current best car
                    print("")
                    best_car, _ = selection(cars)
                    print(best_car.score)
                    print(best_car.nn.biases)
                    print(best_car.nn.weights)
                if event.key == pg.K_h:
                    roads = [Road([50, 300], [100, 250], 2), Road([50, 300], [500, 100], 1), Road([500, 150], [100, 250], 2),
                             Road([50, 150], [500, 100], 3), Road([50, 0], [100, 250], 2), Road([50, 0], [700, 100], 1),
                             Road([650, 0], [100, 550], 4), Road([50, 450], [700, 100], 3)]
                    checkpoints = []
                    for road in roads:
                        checkpoints += road.get_checkpoints()
                if event.key == pg.K_r:  # resets to random initialization
                    win.fill("black")
                    cars = [Car([100, 500], None, None, checkpoints[:]) for _ in range(num)]
                    runs = 0
            if event.type == pg.QUIT:
                best_car, _ = selection(cars)
                print(best_car.nn.weights)
                print(best_car.nn.biases)
                run = False

if __name__ == '__main__':
    main()
