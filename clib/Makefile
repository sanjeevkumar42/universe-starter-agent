all: printscr.so

printscr.so: src/printscr.c
	gcc -shared -O3 -fPIC -Wl,-soname,printscr -o printscr.so src/printscr.c -lX11

clean:
	rm -rf printscr.so
