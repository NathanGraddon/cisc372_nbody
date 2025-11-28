CC = nvcc
FLAGS = -Xcompiler -fopenmp
LIBS = -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	$(CC) $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h compute.h $(ALWAYS_REBUILD)
	$(CC) $(FLAGS) -c $<

compute.o: compute.c config.h vector.h compute.h $(ALWAYS_REBUILD)
	$(CC) -x cu $(FLAGS) -c $<

clean:
	rm -f *.o nbody
