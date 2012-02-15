#include <stdio.h>

int main() {
  unsigned int count;
  for(count = 0; count < 32; count++) {
    printf("%ud\n", ((count & 1) ? count + 31 : count));
  }
  printf("%d\n", 4 & 30);
  int id;
  int cur = 1;
  int width = 32;
  int shift = 0;
  int j;
  int flag;
  while(cur < width) {
    for(id = 0; id < width; id++) {
      flag = 1;
      for(j = 0; j < shift; j++) {
	if((id >> j) & 1) {
	  flag = 0;
	  break;
	}
      }
      if(flag && !((id >> shift) & 1)) {
	printf("%2d ", id);
      }
    }
    printf("\n");
    cur <<= 1;
    shift += 1;
  }
}
