#include <stdio.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
//Compile hint: gcc -shared -O3 -Wall -fPIC -Wl,-soname,printscr -o printscr.so printscr.c -lX11

extern int XDestroyWindow(
    Display*            /* display */,
    Window              /* w */
);


void getScreen(const char*, const int, const int, const int, const int, unsigned char *);
void getScreen(const char* display_name, const int xx,const int yy,const int W, const int H, /*out*/ unsigned char * data)
{
   Display *display = XOpenDisplay(display_name);
   Window root = DefaultRootWindow(display);

   XImage *image = XGetImage(display,root, xx,yy, W,H, AllPlanes, ZPixmap);

   unsigned long red_mask   = image->red_mask;
   unsigned long green_mask = image->green_mask;
   unsigned long blue_mask  = image->blue_mask;
   int x, y;
   int ii = 0;
   for (y = 0; y < H; y++) {
       for (x = 0; x < W; x++) {
         unsigned long pixel = XGetPixel(image,x,y);
         unsigned char blue  = (pixel & blue_mask);
         unsigned char green = (pixel & green_mask) >> 8;
         unsigned char red   = (pixel & red_mask) >> 16;

         data[ii + 2] = blue;
         data[ii + 1] = green;
         data[ii + 0] = red;
         ii += 3;
      }
   }

   XDestroyImage(image);
   XDestroyWindow(display, root);
   XCloseDisplay(display);
}

/*
int main(){

    unsigned char* data = (unsigned char*)malloc(480*640*3);
    getScreen(0, 0, 640, 480, data);
    printf("%d %d \n", data[0], data[1000]);

}
*/