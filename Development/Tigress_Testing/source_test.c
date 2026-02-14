#include <stdio.h>

int add(int a, int b) {
	return a + b;
}

int main(int argc, char**argv) { 
	printf("Hello world!\n");

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			printf("%d + %d = %d\n", i, j, add(i, j));
		}
	}
	
	return 0;
}
