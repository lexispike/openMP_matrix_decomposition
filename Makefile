# compile Q1.cpp
Q1: Q1.cpp
	g++ -fopenmp Q1.cpp -o Q1

# run Q1
run_Q1: Q1
	./Q1

# clean the directory
clean:
	rm -f Q1
