#include <stdio.h>

int main()
{
	while( EOF != fgetc(stdin) )
		;

	return 0;
}
