/*
	convert cvs data to libsvm/svm-light format 
	Updated on Jan 11, 2014 to use strsep() instead of strtok().
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef __USE_BSD 
char *strsep(char **stringp, const char *delim);
#endif

char buf[10000000];
float feature[100000];

int main(int argc, char **argv)
{
	FILE *fp;
	
	if(argc!=2) { fprintf(stderr,"Usage %s filename\n",argv[0]); }
	if((fp=fopen(argv[1],"r"))==NULL)
	{
		fprintf(stderr,"Can't open input file %s\n",argv[1]);
	}
	
	while(fscanf(fp,"%[^\n]\n",buf)==1)
	{
		int i=0,j;
		/*
		char *p=strtok(buf,",");
		
		feature[i++]=atof(p);

		while((p=strtok(NULL,",")))
			feature[i++]=atof(p);
		*/

		char *ptr=buf;
		while (ptr != NULL)
		{
			char *token = strsep(&ptr, ",");
			if (strlen(token) == 0)	
				feature[i++] = 0.0;
			else	
				feature[i++] = atof(token);
		}
		
		//		--i;
		/*
		if ((int) feature[i]==1)
			printf("-1 ");
		else
			printf("+1 ");
		*/
		//		printf("%f ", feature[1]);
		printf("%d ", (int) feature[0]);
		for(j=1;j<i;j++)
			if(feature[j]!=0)
				printf(" %d:%f",j,feature[j]);


		printf("\n");
	}
	return 0;
}

#ifndef __USE_BSD 
/*-
 * Copyright (c) 1990, 1993
 *	The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
char *strsep(char **stringp, const char *delim){
	char *s;
	const char *spanp;
	int c, sc;
	char *tok;

	if ((s = *stringp) == NULL)
		return (NULL);
	for (tok = s;;) {
		c = *s++;
		spanp = delim;
		do {
			if ((sc = *spanp++) == c) {
				if (c == 0)
					s = NULL;
				else
					s[-1] = 0;
				*stringp = s;
				return (tok);
			}
		} while (sc != 0);
	}
}
#endif

